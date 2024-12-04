from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.until_module import PreTrainedModel, AllGather, CrossEn,TripletLoss,Margin2Loss,CrossEn_maxcol,HiTripletLoss
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip_vpt import CLIP, convert_weights
from modules.modeling import CLIP4ClipPreTrainedModel, show_log, update_attr, check_attr


logger = logging.getLogger(__name__)
allgather = AllGather.apply


class XCLIP(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(XCLIP, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))
        show_log(task_config, "\t loss function: {}".format(task_config.loss_func))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders
        
        # frozen the parameter of the visual encoder
        # if hasattr(task_config, "frozen_clip"):
        #     self.frozen_clip = task_config.frozen_clip
        #     show_log(task_config, "\t frozen_clip: {}".format(self.frozen_clip))
        # for param in self.clip.visual.parameters():
        #     param.requires_grad = False 


        # frozen the visual encoder of CLIP
        #################
        # for name, param in self.clip.named_parameters():
        #     # print(name)
        #     if name == "visual.vpt_embedding":
        #         print("#^#"*30)
        #         print( '{} is not frozen'.format(name))
        #         print("#^#"*30)
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # for param in self.clip.transformer.parameters():
        #     param.requires_grad = False
        #################

        # # #frozen the text encoder of CLIP excep the last two layer of text encoder of CLIP
        # for param in self.clip.transformer.resblocks[-2].parameters():
        #     param.requires_grad = True



        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf" or self.sim_header == "seqVptTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)

        if self.sim_header == "seqTransf" or self.sim_header == "seqVptTransf":

            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        # if self.sim_header == "seqVptTransf":
        #     self.transformerVptClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
        #                                            heads=transformer_heads, )

        
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)


        # frozen_seq_transf = True
        frozen_seq_transf=False
        if frozen_seq_transf:
            for param in self.transformerClip.parameters():
                param.requires_grad = False

        num_words = task_config.max_words
        num_frames = task_config.max_frames




        self.loss_func = task_config.loss_func
        # 


        # recommend set True
        self.use_original_clip_for_frame_features = True    

        # for coarse-grained constrast weights
        self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)

        self.global_vp_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)##caz

        # for sentence-level constrast weights
        self.sents_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)###CAZ
        self.sents_vp_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)###CAZ


        # for cross-grained constrast weights
        self.word_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.frame_logit_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)        

        self.word_vp_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.frame_vp_logit_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)

        # for fine-grained constrast weights
        self.local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
        self.local_vp_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)###caz

        self.neg_local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)###caz
        self.frame_mat_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        self.word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.neg_word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)##caz

        self.word_vp_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)

        self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.word_vp_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)

        self.loss_fct = CrossEn()
        self.loss_fct_col = CrossEn_maxcol()
        self.loss_fct_triplet =TripletLoss(margin=0.1)
        self.loss_fct_hitriplet =HiTripletLoss(margin=0.2)
        self.loss_fct_boundary =Margin2Loss(bottommargin=0.1,uppermargin=0.6,
                                           bottommargin_t2t=0.1,uppermargin_t2t=0.3,
                                          measure='cosine',
                                          cost_style='mean')

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, word_ids_aug, word_token_type_ids_aug, word_attention_mask_aug, phrase_ids_aug, phrase_token_type_ids_aug, phrase_attention_mask_aug, video_mask=None):
    # def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        # [bs, 1, dim], [bs, num_words, dim], [bs, num_frames, dim]
        # #### original
        # (sequence_output, seq_features), visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask, 
        #                                                         video, video_mask, shaped=True, video_frame=video_frame)

        ##### for vpt
        (sequence_output, seq_features), visual_output,visual_prompt = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask, 
                                                                video, video_mask, shaped=True,return_vp=True, video_frame=video_frame)

        #### for triplet
        # input_ids_aug = input_ids_aug.view(-1, input_ids_aug.shape[-1])
        # token_type_ids_aug =input_ids_aug.view(-1, token_type_ids_aug.shape[-1])
        # attention_mask_aug = attention_mask_aug.view(-1, token_type_ids_aug.shape[-1])
        # sequence_output_neg, seq_features_neg = self.get_sequence_output(input_ids_aug, token_type_ids_aug, attention_mask_aug, shaped=True) #1102
        neg_sequence_output = []
        neg_seq_features =[]
        for tep_input_ids_aug,tep_token_type_ids_aug,tep_attention_mask_aug in zip( word_ids_aug, word_token_type_ids_aug, word_attention_mask_aug):
            tep_input_ids_aug = tep_input_ids_aug.squeeze(dim=0)
            tep_token_type_ids_aug = tep_token_type_ids_aug.squeeze(dim=0)
            tep_attention_mask_aug = tep_attention_mask_aug.squeeze(dim=0)
            tep_sequence_output_neg,tep_seq_features_neg =self.get_sequence_output(tep_input_ids_aug, tep_token_type_ids_aug, tep_attention_mask_aug, shaped=True)
            # import pdb;pdb.set_trace()
            neg_sequence_output.append(tep_sequence_output_neg)
            neg_seq_features.append(tep_seq_features_neg)

        phrase_neg_sequence_output = []
        phrase_neg_seq_features =[]
        for tep_input_ids_aug,tep_token_type_ids_aug,tep_attention_mask_aug in zip( phrase_ids_aug, phrase_token_type_ids_aug, phrase_attention_mask_aug):
            tep_input_ids_aug = tep_input_ids_aug.squeeze(dim=0)
            tep_token_type_ids_aug = tep_token_type_ids_aug.squeeze(dim=0)
            tep_attention_mask_aug = tep_attention_mask_aug.squeeze(dim=0)
            tep_sequence_output_neg,tep_seq_features_neg =self.get_sequence_output(tep_input_ids_aug, tep_token_type_ids_aug, tep_attention_mask_aug, shaped=True)
            # import pdb;pdb.set_trace()
            phrase_neg_sequence_output.append(tep_sequence_output_neg)
            phrase_neg_seq_features.append(tep_seq_features_neg)
        # import pdb;pdb.set_trace()
    
        loss_t2v = []
        loss_v2t = []
        loss_total = []
        loss_word = []
        loss_phrase = []
      
        if self.training:
            if self.loss_func == "fineGrained" :
                sequence_output_neg, seq_features_neg = self.get_sequence_output(word_ids_aug, word_token_type_ids_aug, word_attention_mask_aug, shaped=True) #1102
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                sim_loss = (sim_loss1 + sim_loss2) / 2
                loss += sim_loss
                
            if self.loss_func == "triplet" :
                sequence_output_neg, seq_features_neg = self.get_sequence_output(word_ids_aug, word_token_type_ids_aug, word_attention_mask_aug, shaped=True)
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                neg_sim_matrix, *_tmp = self.get_similarity_logits(sequence_output_neg, seq_features_neg, visual_output, attention_mask,  video_mask, shaped=True, loose_type=self.loose_type)
                sim_loss_neg = self.loss_fct_triplet(sim_matrix,neg_sim_matrix)
                sim_loss = (sim_loss1 + sim_loss2) / 2 + 0.2*sim_loss_neg
                loss += sim_loss
            
            if self.loss_func == "boundary" :
                sequence_output_neg, seq_features_neg = self.get_sequence_output(word_ids_aug, word_token_type_ids_aug, word_attention_mask_aug, shaped=True)
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)    
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                neg_sim_matrix, *_tmp = self.get_similarity_logits(sequence_output_neg, seq_features_neg, visual_output, attention_mask,  video_mask, shaped=True, loose_type=self.loose_type)
                sim_loss_neg = self.loss_fct_boundary(sim_matrix,neg_sim_matrix)
                sim_loss = (sim_loss1 + sim_loss2) / 2 + 0.2*sim_loss_neg
                loss += sim_loss

            if self.loss_func == "maxcol" :
                #calculate the word level and phrase loss together
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                v2negt_sim_matrix=[]
                for singel_video_feat,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,video_mask,neg_sequence_output,neg_seq_features,word_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix.append(singe_v2t_sim_matrix_semantic.T)  
                v2negt_sim_matrix = torch.cat(v2negt_sim_matrix, dim=0)
                
                sim_loss_neg = self.loss_fct_col(v2negt_sim_matrix)
                v2negt_sim_matrix_phrase=[]
                for singel_video_feat,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,video_mask,phrase_neg_sequence_output,phrase_neg_seq_features,phrase_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix_phrase.append(singe_v2t_sim_matrix_semantic.T)
                v2negt_sim_matrix_phrase = torch.cat(v2negt_sim_matrix_phrase, dim=0)
                sim_loss_neg_phrase = self.loss_fct_col(v2negt_sim_matrix_phrase)
                sim_loss = (sim_loss1 + sim_loss2 ) / 2 + 0.2*sim_loss_neg + 0.2*sim_loss_neg_phrase
                loss += sim_loss
                # loss_total.append(loss.item())
                # loss_t2v.append(sim_loss1.item())
                # loss_v2t.append(sim_loss2.item())
                # loss_word.append(sim_loss_neg.item())
                # loss_phrase.append(sim_loss_neg_phrase.item())
                print("sim_loss1",sim_loss1)
                print("sim_loss2",sim_loss2)
                print("sim_loss_neg",sim_loss_neg)
                print("sim_loss_neg_phrase",sim_loss_neg_phrase)
                print("sim_loss",sim_loss)

                # show the loss curve in a tensorboard


                # import pdb;pdb.set_trace()
                # print("sim_loss1",sim_loss1)
                # print("sim_loss2",sim_loss2)
                # print("sim_loss_neg",sim_loss_neg)
                # print("sim_loss_neg_phrase",sim_loss_neg_phrase)
                # print("sim_loss",sim_loss)

            if self.loss_func == "maxcol_word_vpt" :
                
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)    
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                v2negt_sim_matrix_word=[]
                for singel_video_feat,singel_visual_prompt,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,visual_prompt,video_mask,neg_sequence_output,neg_seq_features,word_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singel_visual_prompt =singel_visual_prompt.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_vpt_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat,singel_visual_prompt, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix_word.append(singe_v2t_sim_matrix_semantic.T)
                #v2negt_sim_matrix[:, 1:] why TypeError: list indices must be integers or slices, not tuple？
                v2negt_sim_matrix_word = torch.cat(v2negt_sim_matrix_word, dim=0)
                sim_loss_neg_word = self.loss_fct_col(v2negt_sim_matrix_word)
                sim_loss = (sim_loss1 + sim_loss2 ) / 2 + 0.2*sim_loss_neg_word
                
                # print('sim_loss_neg_word is:',sim_loss_neg_word)
                loss += sim_loss

            if self.loss_func == "ft_maxcol_word_vpt" :
                
                loss = 0.
                # sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                #                             video_mask, shaped=True, loose_type=self.loose_type)    
                # sim_loss1 = self.loss_fct(sim_matrix)
                # sim_loss2 = self.loss_fct(sim_matrix.T)
                v2negt_sim_matrix_word=[]
                for singel_video_feat,singel_visual_prompt,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,visual_prompt,video_mask,neg_sequence_output,neg_seq_features,word_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singel_visual_prompt =singel_visual_prompt.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_vpt_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat,singel_visual_prompt, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix_word.append(singe_v2t_sim_matrix_semantic.T)
                #v2negt_sim_matrix[:, 1:] why TypeError: list indices must be integers or slices, not tuple？
                v2negt_sim_matrix_word = torch.cat(v2negt_sim_matrix_word, dim=0)
                sim_loss_neg_word = self.loss_fct_col(v2negt_sim_matrix_word)
                sim_loss =sim_loss_neg_word
                
                # print('sim_loss_neg_word is:',sim_loss_neg_word)
                loss += sim_loss


            if self.loss_func == "ft_maxcol_word_vpt_CLS" :
                
                loss = 0.
                # sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                #                             video_mask, shaped=True, loose_type=self.loose_type)    
                # sim_loss1 = self.loss_fct(sim_matrix)
                # sim_loss2 = self.loss_fct(sim_matrix.T)
                v2negt_sim_matrix_word=[]
                for singel_video_feat,singel_visual_prompt,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,visual_prompt,video_mask,neg_sequence_output,neg_seq_features,word_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singel_visual_prompt =singel_visual_prompt.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_vpt_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat,singel_visual_prompt, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix_word.append(singe_v2t_sim_matrix_semantic.T)
                #v2negt_sim_matrix[:, 1:] why TypeError: list indices must be integers or slices, not tuple？
                v2negt_sim_matrix_word = torch.cat(v2negt_sim_matrix_word, dim=0)
                sim_loss_neg_word = self.loss_fct_col(v2negt_sim_matrix_word)
                sim_loss =sim_loss_neg_word
                
                # print('sim_loss_neg_word is:',sim_loss_neg_word)
                loss += sim_loss


            if self.loss_func == "maxcol_word" :
                
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)    
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                v2negt_sim_matrix_word=[]
                for singel_video_feat,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,video_mask,neg_sequence_output,neg_seq_features,word_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix_word.append(singe_v2t_sim_matrix_semantic.T)
                #v2negt_sim_matrix[:, 1:] why TypeError: list indices must be integers or slices, not tuple？
                v2negt_sim_matrix_word = torch.cat(v2negt_sim_matrix_word, dim=0)
                sim_loss_neg_word = self.loss_fct_col(v2negt_sim_matrix_word)
                sim_loss = (sim_loss1 + sim_loss2 ) / 2 + 0.2*sim_loss_neg_word
                
                # print('sim_loss_neg_word is:',sim_loss_neg_word)
                loss += sim_loss

            if self.loss_func == "maxcol_phrase" :
                # sequence_output_neg, seq_features_neg = self.get_sequence_output(phrase_ids_aug, phrase_token_type_ids_aug, phrase_attention_mask_aug, shaped=True)
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)    
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                v2negt_sim_matrix_phrase=[]
                for singel_video_feat,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,video_mask,phrase_neg_sequence_output,phrase_neg_seq_features,phrase_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix_phrase.append(singe_v2t_sim_matrix_semantic.T)
                #v2negt_sim_matrix[:, 1:] why TypeError: list indices must be integers or slices, not tuple？
                v2negt_sim_matrix_phrase = torch.cat(v2negt_sim_matrix_phrase, dim=0)
                sim_loss_neg_phrase = self.loss_fct_col(v2negt_sim_matrix_phrase)
                sim_loss = (sim_loss1 + sim_loss2 ) / 2 + 0.2*sim_loss_neg_phrase
                loss += sim_loss


            if self.loss_func == "maxcol_phrase_vpt" :
                # sequence_output_neg, seq_features_neg = self.get_sequence_output(phrase_ids_aug, phrase_token_type_ids_aug, phrase_attention_mask_aug, shaped=True)
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)    
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                v2negt_sim_matrix_phrase=[]
                for singel_video_feat,singel_visual_prompt,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,visual_prompt,video_mask,phrase_neg_sequence_output,phrase_neg_seq_features,phrase_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singel_visual_prompt =singel_visual_prompt.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_vpt_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat,singel_visual_prompt, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix_phrase.append(singe_v2t_sim_matrix_semantic.T)
                #v2negt_sim_matrix[:, 1:] why TypeError: list indices must be integers or slices, not tuple？
                v2negt_sim_matrix_phrase = torch.cat(v2negt_sim_matrix_phrase, dim=0)
                sim_loss_neg_phrase = self.loss_fct_col(v2negt_sim_matrix_phrase)
                sim_loss = (sim_loss1 + sim_loss2 ) / 2 + 0.2*sim_loss_neg_phrase
                loss += sim_loss

            if self.loss_func == "hierachicalRank" :
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                v2negt_sim_matrix=[]
                for singel_video_feat,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,video_mask,neg_sequence_output,neg_seq_features,word_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix.append(singe_v2t_sim_matrix_semantic.T)

                v2negt_sim_matrix = torch.cat(v2negt_sim_matrix, dim=0)
                sim_loss_neg = self.loss_fct_hitriplet(sim_matrix,v2negt_sim_matrix[:, 1:])

                v2negt_sim_matrix_phrase=[]
                for singel_video_feat,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,video_mask,phrase_neg_sequence_output,phrase_neg_seq_features,phrase_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix_phrase.append(singe_v2t_sim_matrix_semantic.T)
                v2negt_sim_matrix_phrase = torch.cat(v2negt_sim_matrix_phrase, dim=0)
                sim_loss_neg_phrase = self.loss_fct_hitriplet(sim_matrix,v2negt_sim_matrix_phrase[:, 1:])
                sim_loss = (sim_loss1 + sim_loss2 ) / 2 + 0.2*(sim_loss_neg + sim_loss_neg_phrase)
                loss += sim_loss



            if self.loss_func == "hierachicalRank_word" :
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)    
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                v2negt_sim_matrix=[]
                for singel_video_feat,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,video_mask,neg_sequence_output,neg_seq_features,word_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix.append(singe_v2t_sim_matrix_semantic.T)
                #v2negt_sim_matrix[:, 1:] why TypeError: list indices must be integers or slices, not tuple？
                v2negt_sim_matrix = torch.cat(v2negt_sim_matrix, dim=0)
                sim_loss_neg = self.loss_fct_hitriplet(sim_matrix,v2negt_sim_matrix[:, 1:])
                sim_loss = (sim_loss1 + sim_loss2 ) / 2 + 0.2*sim_loss_neg
                loss += sim_loss

            if self.loss_func == "hierachicalRank_phrase" :
                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)    
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)
                ## add phrase level to calculate the loss
                v2negt_sim_matrix_phrase=[]
                for singel_video_feat,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,video_mask,phrase_neg_sequence_output,phrase_neg_seq_features,phrase_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix_phrase.append(singe_v2t_sim_matrix_semantic.T)
                #v2negt_sim_matrix[:, 1:] why TypeError: list indices must be integers or slices, not tuple？       
                v2negt_sim_matrix_phrase = torch.cat(v2negt_sim_matrix_phrase, dim=0)
                sim_loss_neg_phrase = self.loss_fct_hitriplet(sim_matrix,v2negt_sim_matrix_phrase[:, 1:])
                sim_loss = (sim_loss1 + sim_loss2 ) / 2 + 0.2*sim_loss_neg_phrase
                loss += sim_loss

            if self.loss_func == "mix" :

                loss = 0.
                sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)
                sim_loss1 = self.loss_fct(sim_matrix)
                sim_loss2 = self.loss_fct(sim_matrix.T)

                v2negt_sim_matrix=[]
                for singel_video_feat,singel_video_mask,k_neg_sent_feat,k_neg_seq_features,k_neg_attention_mask in zip(visual_output,video_mask,neg_sequence_output,neg_seq_features,word_attention_mask_aug):
                    singel_video_feat =singel_video_feat.unsqueeze(dim=0)
                    singel_video_mask =singel_video_mask.unsqueeze(dim=0)
                    singe_v2t_sim_matrix_semantic,*_tmp  = self.get_similarity_logits(k_neg_sent_feat,k_neg_seq_features, singel_video_feat, k_neg_attention_mask,singel_video_mask,shaped=True, loose_type=self.loose_type)
                    v2negt_sim_matrix.append(singe_v2t_sim_matrix_semantic.T)
                #v2negt_sim_matrix[:, 1:] why TypeError: list indices must be integers or slices, not tuple？
                v2negt_sim_matrix = torch.cat(v2negt_sim_matrix, dim=0)
                sim_loss_neg_hitri = self.loss_fct_hitriplet(sim_matrix,v2negt_sim_matrix[:, 1:])
                # sim_loss = (sim_loss1 + sim_loss2 ) / 2 + 0.2*sim_loss_neg_hitri 
                
                sim_loss_neg_col = self.loss_fct_col(v2negt_sim_matrix)
                sim_loss = (sim_loss1 + sim_loss2 ) / 2 + 0.1*sim_loss_neg_hitri + 0.2*sim_loss_neg_col
                
                loss += sim_loss

            return loss
            # return loss, loss_total, loss_t2v, loss_v2t, loss_word, loss_phrase
            
        else:
            return None
        

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden, seq_features = self.clip.encode_text(input_ids, return_hidden=True)
        sequence_hidden, seq_features = sequence_hidden.float(), seq_features.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden, seq_features

    def get_visual_output(self, video, video_mask, shaped=False, return_vp=False,video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)

        if return_vp:
            visual_hidden,visual_prompt = self.clip.encode_image(video, return_vp=True, video_frame=video_frame)

            visual_hidden = visual_hidden.float()
            visual_prompt = visual_prompt.float()
            visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
            visual_prompt = visual_prompt.view(bs_pair, -1, visual_prompt.size(-1))
            return visual_hidden,visual_prompt
        else:   
            visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
            visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
            return visual_hidden
    
        # return visual_hidden
        # return visual_hidden,visual_prompt_embed

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, return_vp=False,video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output, seq_features = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True) # [bs, 1, dim], [bs, num_words, dim]
        if return_vp:
            visual_output,visual_prompt = self.get_visual_output(video, video_mask, shaped=True, return_vp=True,video_frame=video_frame)                  # [bs, num_frames, dim]
            return (sequence_output, seq_features), visual_output,visual_prompt
        else:
            visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)                  # [bs, num_frames, dim]
            return (sequence_output, seq_features), visual_output

        # return (sequence_output, seq_features), visual_output


    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, seq_features, visual_output, attention_mask, video_mask, sim_header="meanP"):
        """
            sequence_output: CLS token of text       # [bs, 1, dim]
            seq_features: all tokens of text         # [bs, num_words, dim]
            visual_output: all frames of video       # [bs, num_frames, dim]
        """
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            visual_output_original = visual_output
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original

        elif sim_header == "seqTransf" or sim_header == "seqVptTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original


        # video-level visual feature 
        video_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_output = self._mean_pooling_for_similarity_visual(video_output, video_mask)
        video_output = video_output / video_output.norm(dim=-1, keepdim=True)                    # [bs, dim]

        # frame-level visual features       
        if self.use_original_clip_for_frame_features:
            frame_features = visual_output_original / visual_output_original.norm(dim=-1, keepdim=True)                # [bs, num_frames, dim]
        else:
            frame_features = visual_output / visual_output.norm(dim=-1, keepdim=True)                                  # [bs, num_frames, dim]

        # sentence-level textual feature
        sentence_output = sequence_output.squeeze(1)
        sentence_output  = sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]
        
        # word-level textual features
        word_features = seq_features / seq_features.norm(dim=-1, keepdim=True)                   # [bs, num_words, dim]

        logit_scale = self.clip.logit_scale.exp()
        # print('-*-'*20)
        # print('logit_scale is ',logit_scale)
        # print(self.clip.logit_scale)
        # print('-*-'*20)
        # import pdb; pdb.set_trace() 
        if self.training:
            video_output = allgather(video_output, self.task_config)
            frame_features = allgather(frame_features, self.task_config)
            sentence_output = allgather(sentence_output, self.task_config)
            word_features = allgather(word_features, self.task_config)
            torch.distributed.barrier()

        # video-sentence score 
        video_sentence_logits = logit_scale * torch.matmul(torch.matmul(sentence_output, self.global_mat_weight), video_output.t())

        # video-word score
        video_word_logits = logit_scale * torch.sum(torch.matmul(word_features, video_output.t()) \
            * torch.matmul(torch.softmax(torch.matmul(word_features, video_output.t()) / 1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1)

        # sentence-frame score 
        sentence_frame_logits = logit_scale * torch.sum(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) \
            * torch.matmul(torch.softmax(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()

        # frame-word score
        frame_word_logits = logit_scale * self._attenion_over_fine_grained_sim_matrix(word_features, frame_features)

        logits = (video_sentence_logits + video_word_logits + sentence_frame_logits + frame_word_logits) / 4

        return logits



    def _loose_vpt_similarity(self, sequence_output, seq_features, visual_output,visual_prompt, attention_mask, video_mask, sim_header="meanP"):
        """
            sequence_output: CLS token of text       # [bs, 1, dim]
            seq_features: all tokens of text         # [bs, num_words, dim]
            visual_output: all frames of video       # [bs, num_frames, dim]
        """
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        visual_prompt = visual_prompt.contiguous()

         
        # Sequential type: Transformer Encoder with Visual Prompt
        visual_output_original = visual_output
        visual_prompt_original = visual_prompt

        seq_length = visual_output.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)

        visual_output = visual_output + frame_position_embeddings
        visual_prompt = visual_prompt + frame_position_embeddings#caz 20240609  add


        
        extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        # import pdb;pdb.set_trace()
        visual_output = visual_output.permute(1, 0, 2)
        visual_prompt = visual_prompt.permute(1, 0, 2)

        visual_output = self.transformerClip(visual_output, extended_video_mask)
        visual_prompt = self.transformerClip(visual_prompt, extended_video_mask)

        visual_output = visual_output.permute(1, 0, 2)
        visual_prompt = visual_prompt.permute(1, 0, 2)

        visual_output = visual_output + visual_output_original
        visual_prompt = visual_prompt + visual_prompt_original




        # video-level visual feature 
        video_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_output = self._mean_pooling_for_similarity_visual(video_output, video_mask)
        video_output = video_output / video_output.norm(dim=-1, keepdim=True)                    # [bs, dim]

        #video-level visual prompt feature
        visual_prompt_output = visual_prompt / visual_prompt.norm(dim=-1, keepdim=True)
        visual_prompt_output = self._mean_pooling_for_similarity_visual(visual_prompt_output, video_mask)
        visual_prompt_output = visual_prompt_output / visual_prompt_output.norm(dim=-1, keepdim=True)                    # [bs, dim]

        # frame-level visual features       
        if self.use_original_clip_for_frame_features:
            frame_features = visual_output_original / visual_output_original.norm(dim=-1, keepdim=True)                # [bs, num_frames, dim]
        else:
            frame_features = visual_output / visual_output.norm(dim=-1, keepdim=True)                                  # [bs, num_frames, dim]

        # frame-level visual prompt features
        # if self.use_original_clip_for_frame_features:
        #     frame_prompt_features = visual_prompt_original / visual_prompt_original.norm(dim=-1, keepdim=True)                # [bs, num_frames, dim]
        # else:
        frame_prompt_features = visual_prompt / visual_prompt.norm(dim=-1, keepdim=True)                                  # [bs, num_frames, dim] 

        # sentence-level textual feature
        sentence_output = sequence_output.squeeze(1)
        sentence_output  = sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]
        
        # word-level textual features
        word_features = seq_features / seq_features.norm(dim=-1, keepdim=True)                   # [bs, num_words, dim]

        logit_scale = self.clip.logit_scale.exp()
   
        if self.training:
            video_output = allgather(video_output, self.task_config)
            frame_features = allgather(frame_features, self.task_config)
            sentence_output = allgather(sentence_output, self.task_config)
            word_features = allgather(word_features, self.task_config)

            visual_prompt_output = allgather(visual_prompt_output, self.task_config)
            frame_prompt_features = allgather(frame_prompt_features, self.task_config)

            torch.distributed.barrier()

        # video-sentence score 
        # video_sentence_logits = logit_scale * torch.matmul(torch.matmul(sentence_output, self.global_mat_weight), video_output.t())
        # visual_prompt-sentence score
        visual_prompt_sentence_logits = logit_scale * torch.matmul(torch.matmul(sentence_output, self.global_vp_mat_weight), visual_prompt_output.t())

        # visual_prompt-word score
        #word_features torch.Size([16, 32, 512])
        # video_word_logits = logit_scale * torch.sum(torch.matmul(word_features, video_output.t()) \
        #     * torch.matmul(torch.softmax(torch.matmul(word_features, video_output.t()) / 1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1)

        # import pdb;pdb.set_trace()

        # visual_prompt_word_logits = logit_scale * torch.sum(torch.matmul(word_features, visual_prompt_output.t()) * torch.matmul(torch.softmax(torch.matmul(word_features, visual_prompt_output.t()) / 1e-2, dim=1).permute(0,2,1), self.word_vp_logit_weight).permute(0,2,1), dim=1)

        visual_prompt_word_logits = logit_scale * torch.sum(torch.matmul(word_features, visual_prompt_output.t()) * torch.softmax(torch.matmul(word_features, visual_prompt_output.t())/ 1e-2, dim=1), dim=1)


        # sentence-frame score 
        # sentence_frame_logits = logit_scale * torch.sum(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) \
        #     * torch.matmul(torch.softmax(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()

        # sentence-_prompt_frame score 
        sentence_prompt_frame_logits = logit_scale * torch.sum(torch.matmul(sentence_output, frame_prompt_features.permute(0, 2, 1)) \
            * torch.matmul(torch.softmax(torch.matmul(sentence_output, frame_prompt_features.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()

        # frame-word score
        # frame_word_logits = logit_scale * self._attenion_over_fine_grained_sim_matrix(word_features, frame_features)

        frame_prompt_word_logits = logit_scale * self._attenion_over_vp_fine_grained_sim_matrix(word_features, frame_prompt_features)


        # logits = (video_sentence_logits + video_word_logits + sentence_frame_logits + frame_word_logits) / 4

        logits = (visual_prompt_sentence_logits + visual_prompt_word_logits + sentence_prompt_frame_logits +frame_prompt_word_logits) / 4

        return logits



        

        pass
    

    def _loose_t2t_similarity(self, neg_sequence_output, sequence_output, neg_seq_features, seq_features):## 考虑让文本也forward 一个transformer
        """
         
            neg_seq_features: all tokens of text     # [bs, num_words, dim]
            seq_features: all tokens of text         # [bs, num_words, dim]
            visual_output: all frames of video       # [bs, num_frames, dim]
        """
        sequence_output,  neg_sequence_output = sequence_output.contiguous(),  neg_sequence_output.contiguous()
# 

        # sentence-level textual feature
        sentence_output = sequence_output.squeeze(1)
        sentence_output  = sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]
        neg_sentence_output = neg_sequence_output.squeeze(1)
        neg_sentence_output  = neg_sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]
        
        
        # word-level textual features
        word_features = seq_features / seq_features.norm(dim=-1, keepdim=True)                   # [bs, num_words, dim]


        # word-level hard negative textual features
        neg_word_features = neg_seq_features / neg_seq_features.norm(dim=-1, keepdim=True)                   # [bs, num_words, dim]

        logit_scale = self.clip.logit_scale.exp()

        if self.training:
            sentence_output = allgather(sentence_output, self.task_config)
            neg_sentence_output = allgather(neg_sentence_output, self.task_config)
            word_features = allgather(word_features, self.task_config)
            neg_word_features = allgather(neg_word_features, self.task_config)
            torch.distributed.barrier()

        # video-sentence score 
        neg_sentence_logits = logit_scale * torch.matmul(torch.matmul(sentence_output, self.sents_mat_weight), neg_sentence_output.t())

        # frame-word score
        neg_word_logits = logit_scale * self._attenion_over_neg_word_level_sim_matrix(word_features, neg_word_features)

        return (neg_sentence_logits+ neg_word_logits) / 2

        # logits = (neg_sentence_logits+ neg_word_logits) / 2

        # return logits


    def _attenion_over_neg_word_level_sim_matrix(self, word_features, neg_word_features):
        bs_negtext, num_neg_words, dim_neg_text = neg_word_features.shape
        bs_text, num_words, dim_text = word_features.shape
        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.neg_local_mat_weight), neg_word_features.view(-1, dim_neg_text).t()).view(bs_text, num_words, bs_negtext, num_neg_words)  # [bs_text, num_words, bs_video, num_frames]

        neg_word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
      

        neg_sent2sent_logits = torch.sum(torch.matmul(torch.softmax(neg_word_level_logit/1e-2, dim=-1),self.neg_word_mat_weight) * neg_word_level_logit, dim=-1)                                # [bs_text, bs_video]
    

        return neg_sent2sent_logits 




    def _attenion_over_fine_grained_sim_matrix(self, word_features, frame_features):
        bs_video, num_frames, dim_video = frame_features.shape
        bs_text, num_words, dim_text = word_features.shape
        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.local_mat_weight), frame_features.view(-1, dim_video).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]

        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        frame_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_video]

        sent2frame_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.frame_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        video2word_logits = torch.sum(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        return (sent2frame_logits + video2word_logits) / 2

    def _attenion_over_vp_fine_grained_sim_matrix(self, word_features, frame_prompt_features):
        bs_video, num_frames, dim_video = frame_prompt_features.shape
        bs_text, num_words, dim_text = word_features.shape
        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.local_vp_mat_weight), frame_prompt_features.view(-1, dim_video).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]

        # word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_vp_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        word_level_logit = torch.sum(torch.softmax(fine_grained_sim_scores/1e-2, dim=1) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        frame_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_video]

        sent2frame_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.frame_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        
        # video2word_logits = torch.sum(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_vp_mat_weight2).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        video2word_logits = torch.sum(torch.softmax(frame_level_logit/1e-2, dim=1)* frame_level_logit, dim=1)  # [bs_text, bs_video]

        return (sent2frame_logits + video2word_logits) / 2


        # pass

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, seq_features, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf", "seqVptTransf"]
        
            retrieve_logits = self._loose_similarity(sequence_output, seq_features, visual_output, attention_mask, video_mask, sim_header=self.sim_header)

        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction


    def get_vpt_similarity_logits(self, sequence_output, seq_features, visual_output,visual_prompt, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf",  "seqVptTransf"]
            if self.sim_header in ["seqVptTransf"]:
                retrieve_logits = self._loose_vpt_similarity(sequence_output, seq_features, visual_output,visual_prompt, attention_mask, video_mask, sim_header=self.sim_header)
            else:
                retrieve_logits = self._loose_similarity(sequence_output, seq_features, visual_output, attention_mask, video_mask, sim_header=self.sim_header)

            
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction
    
    def get_t2t_similarity_logits(self, neg_sequence_output, sequence_output, neg_seq_features, seq_features):
   
        t2t_retrieve_logits = self._loose_t2t_similarity(neg_sequence_output, sequence_output, neg_seq_features, seq_features)

        return t2t_retrieve_logits