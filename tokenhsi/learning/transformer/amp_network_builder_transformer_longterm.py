from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch.network_builder import NetworkBuilder

from learning.amp_network_builder import AMPBuilder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.transformer.amp_network_builder_transformer import AMPTransformerMultiTaskBuilder

from utils.torch_utils import load_checkpoint

import torch
import torch.nn as nn

class AMPTransformerMultiTaskLongTermTaskCompletionBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPTransformerMultiTaskLongTermTaskCompletionBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPTransformerMultiTaskBuilder.UnifiedNetworkClass):

        def __init__(self, params, **kwargs):

            assert params["network_structure_id"] == 2, "only support transformer now"

            # whether we add internal adaptation layers after the extra feats
            self.use_internal_adaptation = params.get("use_internal_adaptation", False)
            self.apply_adapter_on_actions = params.get("apply_adapter_on_actions", False)

            if self.apply_adapter_on_actions:
                assert self.use_internal_adaptation

            super().__init__(params, **kwargs)

            # Which tasks does the current plan support. For supported tasks, we enable the gradient descent function of the encoder.
            self.learnable_task_names = self.multi_task_info["plan_supported_task"]

            # Variables related to extra height obs
            self.has_extra = self.multi_task_info["has_extra"]
            if self.has_extra:
                self.extra_each_subtask_obs_size = self.multi_task_info["extra_each_subtask_obs_size"]
                self.extra_each_subtask_obs_indx = self.multi_task_info["extra_each_subtask_obs_indx"]

            # Because the onehot code is in the middle of the obs returned by env, such an index is needed
            self.task_obs_onehot_indx = self.multi_task_info["onehot_indx"]

            ckp = load_checkpoint(kwargs["hrl_checkpoint"], device=self.device)
            state_dict_loaded = ckp["model"]

            ##### self token
            key = "self_encoder"
            any_loaded = False
            state_dict = self.self_encoder.state_dict()
            for k in state_dict_loaded.keys():
                if key in k:
                    any_loaded = True
                    pnn_dict_key = k.split(key + ".")[1]
                    state_dict[pnn_dict_key].copy_(state_dict_loaded[k])
            assert any_loaded, "No parameters loaded!!! You are using wrong base models!!!"

            self.self_encoder.requires_grad_(False)
            self.self_encoder.eval()

            ##### multi task tokens
            state_dict_loaded_task_encoders = {}
            for k in state_dict_loaded.keys():
                if "task_encoder" in k:
                    idx = int(k.split(".")[2])
                    if idx not in state_dict_loaded_task_encoders.keys():
                        state_dict_loaded_task_encoders[idx] = []
                    state_dict_loaded_task_encoders[idx].append(state_dict_loaded[k])

            self.basic_num_tasks = len(state_dict_loaded_task_encoders.keys())
            self.basic_task_obs_each_size = [v[0].shape[-1] for k, v in state_dict_loaded_task_encoders.items()]
            self.basic_task_obs_tota_size = sum(self.basic_task_obs_each_size)
            self.basic_task_obs_each_indx = torch.cumsum(torch.tensor([0] + self.basic_task_obs_each_size), dim=0).to(self.device)

            self.basic_task_running_mean_std_needed = nn.ModuleList()
            self.basic_self_running_mean_std = RunningMeanStd(self.self_obs_size)
            self.basic_self_running_mean_std.running_mean = ckp["running_mean_std"]["running_mean"][:self.self_obs_size].clone()
            self.basic_self_running_mean_std.running_var = ckp["running_mean_std"]["running_var"][:self.self_obs_size].clone()
            self.basic_self_running_mean_std.count = ckp["running_mean_std"]["count"].clone()
            
            for i, s in enumerate(self.task_obs_each_size, 0): # init all task encoders
                if s in self.basic_task_obs_each_size:
                    id = self.basic_task_obs_each_size.index(s)

                    rms = RunningMeanStd(s)
                    rms.running_mean = ckp["running_mean_std"]["running_mean"][self.self_obs_size:][self.basic_task_obs_each_indx[id]:self.basic_task_obs_each_indx[id + 1]].clone()
                    rms.running_var = ckp["running_mean_std"]["running_var"][self.self_obs_size:][self.basic_task_obs_each_indx[id]:self.basic_task_obs_each_indx[id + 1]].clone()
                    rms.count = ckp["running_mean_std"]["count"].clone()

                    self.basic_task_running_mean_std_needed.append(rms)
                
                    key = "task_encoder.{}".format(id)
                    any_loaded = False
                    state_dict = self.task_encoder[i].state_dict()
                    for k in state_dict_loaded.keys():
                        if key in k:
                            any_loaded = True
                            pnn_dict_key = k.split(key + ".")[1]
                            state_dict[pnn_dict_key].copy_(state_dict_loaded[k])
                    assert any_loaded, "No parameters loaded!!! You are using wrong base models!!!"

                    curr_task_encoder_name = self.each_subtask_names[i]
                    if curr_task_encoder_name not in self.learnable_task_names:
                        self.task_encoder[i].requires_grad_(False)
                        self.task_encoder[i].eval()

                else:
                    raise NotImplementedError
            
            ##### extra output token
            self.weight_token.requires_grad_(False)
            self.weight_token.copy_(state_dict_loaded["a2c_network.weight_token"])

            ##### backbone
            key = "transformer_encoder"
            any_loaded = False
            state_dict = self.transformer_encoder.state_dict()
            for k in state_dict_loaded.keys():
                if key in k:
                    any_loaded = True
                    pnn_dict_key = k.split(key + ".")[1]
                    state_dict[pnn_dict_key].copy_(state_dict_loaded[k])
            assert any_loaded, "No parameters loaded!!! You are using wrong base models!!!"

            self.transformer_encoder.requires_grad_(False)
            self.transformer_encoder.eval()
            
            ##### extra mlp
            key = "composer"
            any_loaded = False
            state_dict = self.composer.state_dict()
            for k in state_dict_loaded.keys():
                if key in k:
                    pnn_dict_key = k.split(key + ".")[1]
                    state_dict[pnn_dict_key].copy_(state_dict_loaded[k])

            self.composer.requires_grad_(False)
            self.composer.eval()

            #################### trainable networks

            num_features = self.weight_token.shape[-1]

            if self.has_extra:
                ######## the extra trainable task tokenizer
                ### load extra module for extra obs
                mlp_args = {
                    'input_size' : self.extra_each_subtask_obs_size, 
                    'units' : params["new_input_tokenizer_units"] + [num_features], 
                    'activation' : self.activation, 
                    'norm_func_name' : self.normalization,
                    'dense_func' : torch.nn.Linear,
                    'd2rl' : self.is_d2rl,
                    'norm_only_first_layer' : self.norm_only_first_layer
                }
                self.extra_encoder = self._build_mlp(**mlp_args)

                for m in self.extra_encoder.modules(): # zero init
                    if isinstance(m, nn.Linear):
                        # torch.nn.init.zeros_(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)

            if self.use_internal_adaptation:
                plugin_units = params["plugin_units"]
                if self.apply_adapter_on_actions:
                    plugin_units.append(self.dof_action_size)
                mlp_args = {
                    'input_size' : num_features,
                    'units' : plugin_units,
                    'activation' : "None", # [don't use any activation func] 
                    'norm_func_name' : self.normalization,
                    'dense_func' : torch.nn.Linear,
                    'd2rl' : self.is_d2rl,
                    'norm_only_first_layer' : self.norm_only_first_layer
                }
                self.internal_adapt_mlp = self._build_mlp(**mlp_args)

                for m in self.internal_adapt_mlp.modules(): # zero init
                    if isinstance(m, nn.Linear):
                        torch.nn.init.zeros_(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)

            return
        
        def _eval_Transformer(self, obs, not_normalized_obs):

            B = obs.shape[0]
            
            # not_normalized_obs >> self_token
            self.basic_self_running_mean_std.training = False
            self_obs = self.basic_self_running_mean_std(not_normalized_obs[..., :self.self_obs_size])
            self_token = self.self_encoder(self_obs).unsqueeze(1) # (B, 1, num_feats)

            # not_normalized_obs >> relevant task tokens
            task_token = []
            for i in range(self.task_obs_onehot_size):
                curr_task_obs = not_normalized_obs[..., self.self_obs_size:][..., self.task_obs_each_indx[i]:self.task_obs_each_indx[i + 1]]
                if self.each_subtask_names[i] not in self.learnable_task_names:
                    self.basic_task_running_mean_std_needed[i].training = False
                curr_task_obs = self.basic_task_running_mean_std_needed[i](curr_task_obs)
                task_token.append(self.task_encoder[i](curr_task_obs))
            task_token = torch.stack(task_token, dim=1)

            # [1, 1, num_feats] -> [B, 1, num_feats]
            weight_token = self.weight_token.expand(B, -1, -1)

            x = torch.cat((weight_token, self_token, task_token), dim=1)

            # compute key padding mask

            src_key_padding_mask = torch.ones((B, x.shape[1]), dtype=torch.bool, device=x.device) # init to all True
            src_key_padding_mask[:, [0, 1]] = False

            task_obs_onehot_idx = not_normalized_obs[..., self.self_obs_size:][..., self.task_obs_onehot_indx[0]:self.task_obs_onehot_indx[1]].max(dim=-1)[1] + 2
            task_obs_onehot_idx_mask = nn.functional.one_hot(task_obs_onehot_idx, num_classes=self.task_obs_onehot_size + 2).bool()
            
            src_key_padding_mask[task_obs_onehot_idx_mask] = False

            if self.has_extra:
                extra_token = self.extra_encoder(obs[..., self.self_obs_size:][..., self.extra_each_subtask_obs_indx[0]:self.extra_each_subtask_obs_indx[1]])
                extra_token = extra_token.unsqueeze(1)

                x = torch.cat((x, extra_token), dim=1)

                src_key_padding_mask = torch.cat([src_key_padding_mask, torch.zeros((src_key_padding_mask.shape[0], 1), dtype=torch.bool, device=self.device)], dim=-1)

            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

            embeddings = x[:, 0]

            if self.use_internal_adaptation:

                for j, op in enumerate(self.composer.actors[:-1]):
                    embed = self.internal_adapt_mlp[j]
                    if isinstance(embed, torch.nn.Identity):
                        embeddings = op(embeddings)
                    else:
                        embeddings = op(embeddings) + embed(embeddings)
            
                if self.apply_adapter_on_actions:
                    action = self.composer.act(self.composer.actors[-1](embeddings) + self.internal_adapt_mlp[-2](embeddings)) # self.internal_adapt_mlp[-1] is Identity
                else:
                    action = self.composer.act(self.composer.actors[-1](embeddings))

            else:

                meta_action = self.composer(embeddings)
                action = meta_action
            
            return action
