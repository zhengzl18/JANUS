import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import transformers
from datautils import get_knowledge_data
from peft.peft_model import PeftModel
from safetensors.torch import load_file, save_file
from torch.nn import Linear
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100
torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(instance["input_ids"].squeeze(0)) for instance in instances]
        labels = [torch.tensor(instance["labels"].squeeze(0)) for instance in instances]

        lengths = [len(seq) for seq in input_ids]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, l in enumerate(lengths):
            attention_mask[i, :l] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    

class MultiStepAdaptiveRectifier:
    def __init__(
        self,
        original_model,
        finetuned_model,
        target_layers: Sequence[str],
        cache_dir: str,
        r_jac_approx: int = 32,
        delta_threshold: float = 0.95,
        beta: float = 0.7,
        chunk_size: int = 56
    ):
        self.r_jac_approx = r_jac_approx
        self.chunk_size = chunk_size
        self.device = torch.device("cuda:0")
        
        self.model = original_model.to(self.device)
        self.model.gradient_checkpointing_enable()
        finetuned_model = finetuned_model.to(self.device)
        self.target_layers = target_layers
        
        self.delta_w_dict = {}
        for name in target_layers:
            self.delta_w_dict[name] = (
                finetuned_model.get_submodule(name).weight.data - self.model.get_submodule(name).weight.data
            ).clone().cpu()

        torch.cuda.empty_cache()
        self.delta_threshold = delta_threshold
        self.beta = beta
        
        self.jac_dir_old = f"{cache_dir}/jac_old"
        self.jac_dir_new = f"{cache_dir}/jac_new"

        self.Gram_old = None
        self.P_old = None
        self.Gram_new = None
        self.P_new = None
        self.alpha = 1.0

    @torch.no_grad()
    def _svd_compress_features(self, a, b, r=32):
        m, T, d_in = a.shape
        _, _, d_out = b.shape
        actual_r = min(r, T, d_in, d_out)  
        
        # Shape: Q_b: (m, d_out, k_b), k_b = \min(d_out, T)
        #        R_b: (m, k_b, T)
        Q_b, R_b = torch.linalg.qr(b.transpose(1, 2))
        # Shape: Q_a: (m, d_in, k_a), k_a = \min(d_in, T)
        #        R_a: (m, k_a, T)
        Q_a, R_a = torch.linalg.qr(a.transpose(1, 2))
        M = torch.bmm(R_b, R_a.transpose(1, 2))  # Shape: (m, k_b, k_a)
        # Shape: U_M: (m, k_b, k_b), 
        #        S: (m, \min(k_a, k_b)), 
        #        V_M_h: (m, k_a, k_a)
        U_M, S, V_M_h = torch.linalg.svd(M)
        
        U_M_r = U_M[:, :, :actual_r]                
        S_r = S[:, :actual_r]                      
        V_M_r = V_M_h.transpose(1, 2)[:, :, :actual_r] 
        S_sqrt = torch.diag_embed(torch.sqrt(S_r))
        
        new_b_T = torch.bmm(Q_b, torch.bmm(U_M_r, S_sqrt)).transpose(1, 2)  # Shape: (m, r, d_out)
        new_a_T = torch.bmm(Q_a, torch.bmm(V_M_r, S_sqrt)).transpose(1, 2)  # Shape: (m, r, d_in)
        
        if actual_r < r:
            pad_a = torch.zeros(m, r - actual_r, d_in, device=a.device, dtype=a.dtype)
            pad_b = torch.zeros(m, r - actual_r, d_out, device=b.device, dtype=b.dtype)
            new_a_T = torch.cat([new_a_T, pad_a], dim=1)
            new_b_T = torch.cat([new_b_T, pad_b], dim=1)
            
        return new_a_T.to(torch.float16), new_b_T.to(torch.float16)

    def _get_projection_components(self, dataloader, target_jac_dir, for_landing_point=False):
        print("Calculating Jacobian and projection components...")

        Gram = {}
        #* Projection correction term P for each layer
        P = {}

        all_modules = [(name, self.model.get_submodule(name)) for name in self.target_layers]
        module_chunks = [
            all_modules[i:i + self.chunk_size] 
            for i in range(0, len(all_modules), self.chunk_size)
        ]

        for name, module in all_modules:
            module.weight.requires_grad = False
        
        for chunk_idx, chunk in enumerate(module_chunks):
            print(f"\n=== Processing Layer Chunk {chunk_idx + 1}/{len(module_chunks)} ===")
            
            jac_w_cache = {name: {'a': [], 'b': []} for name, _ in chunk}
            hooks = []
            
            for name, module in chunk:
                module.weight.requires_grad = True
                
                def make_hooks(layer_name):
                    fwd_activation_cache = []
                    def fwd_hook(mod, inp, out):
                        fwd_activation_cache.append(inp[0].detach())
                        
                    def bwd_hook(mod, grad_input, grad_output):
                        bwd_activation = grad_output[0].detach()  # Shape: (B, T, d_out)
                        fwd_activation = fwd_activation_cache[-1]  # Shape: (B, T, d_in)
                        fwd_activation_cache.clear()
                        
                        jac_w_a, jac_w_b = self._svd_compress_features(
                            fwd_activation.float(), 
                            bwd_activation.float(), 
                            r=self.r_jac_approx
                        )  
                        jac_w_a, jac_w_b = jac_w_a.contiguous(), jac_w_b.contiguous()
                        jac_w_cache[layer_name]['a'].append(jac_w_a.cpu())
                        jac_w_cache[layer_name]['b'].append(jac_w_b.cpu())
                        
                    return fwd_hook, bwd_hook
                    
                f_hook, b_hook = make_hooks(name)
                hooks.append(module.register_forward_hook(f_hook))
                hooks.append(module.register_full_backward_hook(b_hook))

            print("Calculating Jacobian ...")
            count = 0
            for batch_inputs in dataloader:
                count += 1
                if count % 10 == 0:
                    print(f"  Batch {count}/{len(dataloader)}...")
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                self.model.zero_grad(set_to_none=True)
                outputs = self.model(**batch_inputs)
                labels = batch_inputs["labels"]
                valid_token_num = (labels != IGNORE_INDEX).sum().item()
                loss = outputs.loss * valid_token_num 
                loss.backward()
                self.model.zero_grad(set_to_none=True)

            for h in hooks:
                h.remove()
            for name, module in chunk:
                module.weight.requires_grad = False

            for name, _ in chunk:
                jac_w_a = torch.cat(jac_w_cache[name]['a'], dim=0)
                jac_w_b = torch.cat(jac_w_cache[name]['b'], dim=0)
                
                file_path = f"{target_jac_dir}/{name.replace('.', '-')}.safetensors"
                save_file({"a": jac_w_a, "b": jac_w_b}, file_path)
                
                jac_w_a = jac_w_a.to(self.device)
                jac_w_b = jac_w_b.to(self.device)
                m, r = jac_w_a.shape[0], jac_w_a.shape[1]
                
                jac_w_a_flat = jac_w_a.view(m * r, -1).double()
                jac_w_b_flat = jac_w_b.view(m * r, -1).double()
                
                Ka = jac_w_a_flat @ jac_w_a_flat.T 
                Kb = jac_w_b_flat @ jac_w_b_flat.T
                
                #* G = J @ J^T
                G_local = (Ka * Kb).view(m, r, m, r).sum(dim=(1, 3))  # Shape: (m, m)
                Gram[name] = G_local.cpu()
                
                #* y = J @ delta_w
                if for_landing_point:
                    delta_w = self.delta_w_dict[name].to(self.device, dtype=torch.float64, non_blocking=True)
                    P_old = self.P_old[name].to(self.device, dtype=torch.float64, non_blocking=True)
                    delta_w = (1 - self.alpha) * delta_w + self.alpha * P_old
                else:
                    delta_w = self.delta_w_dict[name].to(self.device, dtype=torch.float64, non_blocking=True)
                Jdw_local = ((jac_w_a_flat @ delta_w.T) * jac_w_b_flat).view(m, r, -1).sum(dim=(1, 2)).cpu()  # Shape: (m,)

                #* P = J^T @ G^-1 @ J @ delta_w
                G_inv = torch.linalg.pinv(Gram[name].double(), rcond=1e-10)
                v = (G_inv @ Jdw_local.double()).to(device=self.device)
                P_local = torch.einsum(
                    'm, mto, mti -> oi', 
                    v, jac_w_b.double(), jac_w_a.double()
                )  # Shape: (d_out, d_in)
                P[name] = P_local.cpu()
                
                del jac_w_cache[name]
                del jac_w_a, jac_w_b, jac_w_a_flat, jac_w_b_flat, Ka, Kb
                torch.cuda.empty_cache()
        
        return Gram, P

    @torch.no_grad()
    def _compute_principal_angle(self, jac_dir_old, jac_dir_new, names, Gram_old, Gram_new, eps=1e-10):
        def compute_mean_cosine(Gram1, Gram2, cross_Gram, eps=1e-10):
            eig0, L0 = torch.linalg.eigh(Gram1.double())
            eig1, L1 = torch.linalg.eigh(Gram2.double())

            mask0 = eig0 > eps
            eig0_inv_sqrt = torch.zeros_like(eig0)
            eig0_inv_sqrt[mask0] = 1.0 / torch.sqrt(eig0[mask0])
            S0 = L0 * eig0_inv_sqrt.unsqueeze(0)  
            
            mask1 = eig1 > eps
            eig1_inv_sqrt = torch.zeros_like(eig1)
            eig1_inv_sqrt[mask1] = 1.0 / torch.sqrt(eig1[mask1])
            S1 = L1 * eig1_inv_sqrt.unsqueeze(0)

            M = S0.T @ cross_Gram @ S1
            cosines = torch.linalg.svdvals(M)
            cosines = torch.clamp(cosines, min=0.0, max=1.0)
            return cosines.mean().item()

        cosines_dict = {}
        for name in names:
            print(f"Computing cross term for {name}...")
            
            file_old = f"{jac_dir_old}/{name.replace('.', '-')}.safetensors"
            file_new = f"{jac_dir_new}/{name.replace('.', '-')}.safetensors"
            
            tensors_0 = load_file(file_old)
            tensors_1 = load_file(file_new)
            
            jac_w_a_0 = tensors_0["a"].to(self.device)  # Shape: (m, r, i)
            jac_w_b_0 = tensors_0["b"].to(self.device)  # Shape: (m, r, o)
            jac_w_a_1 = tensors_1["a"].to(self.device)  # Shape: (m, r, i)
            jac_w_b_1 = tensors_1["b"].to(self.device)  # Shape: (m, r, o)
            
            #* G_{cross} = J1 @ J2^T
            AA_cross = torch.einsum('mti, nsi -> mnts', jac_w_a_0.double(), jac_w_a_1.double())  # Shape: (m, m, r, r)
            BB_cross = torch.einsum('mto, nso -> mnts', jac_w_b_0.double(), jac_w_b_1.double())  # Shape: (m, m, r, r)
            Cross_Gram_local = (AA_cross * BB_cross).sum(dim=(2, 3)).cpu()  # Shape: (m, m)
            
            del jac_w_a_0, jac_w_b_0, jac_w_a_1, jac_w_b_1, AA_cross, BB_cross, tensors_0, tensors_1
            torch.cuda.empty_cache()

            cosines_dict[name] = compute_mean_cosine(Gram_old[name], Gram_new[name], Cross_Gram_local, eps=eps)
            print(f"  {name} mean cosine of principal angles={cosines_dict[name]:.4f}")

        mean_cosine = sum(cosines_dict.values()) / len(cosines_dict)
        return mean_cosine

    def multi_step_adaptive_rectification(self, dataloader):
        print("Clearing old Jacobian cache and preparing directories...")
        shutil.rmtree(self.jac_dir_old, ignore_errors=True)
        shutil.rmtree(self.jac_dir_new, ignore_errors=True)
        os.makedirs(self.jac_dir_old, exist_ok=True)
        os.makedirs(self.jac_dir_new, exist_ok=True)
        
        self.Gram_old, self.P_old = self._get_projection_components(dataloader, self.jac_dir_old)
        
        while True:
            #* Searching for proper alpha with rollback mechanism, starting with a full step (alpha=1.0)
            self.alpha = 1.0
            while True:
                #* Applying projected update with the step length of alpha to the model weights
                for name in self.target_layers:
                    module = self.model.get_submodule(name)
                    delta_w = self.delta_w_dict[name].to(self.device)
                    P = self.P_old[name].to(self.device)
                    
                    delta_w_projed = self.alpha * (delta_w - P)
                    updated_weight = module.weight.data + delta_w_projed
                    module.weight.data.copy_(updated_weight)
                
                #* Calculating new global components with the updated model
                self.Gram_new, self.P_new = self._get_projection_components(dataloader, self.jac_dir_new, for_landing_point=True)
                
                mean_cos = self._compute_principal_angle(
                    self.jac_dir_old, 
                    self.jac_dir_new, 
                    self.target_layers, 
                    self.Gram_old, 
                    self.Gram_new
                )
                print(f"  [Inner] alpha={self.alpha:.4f}, global_mean_cos={mean_cos:.4f}")
                
                if mean_cos > self.delta_threshold:
                    print(f"Global projection converged with alpha={self.alpha:.4f} and mean_cos={mean_cos:.4f}.")            
                    break
                else:
                    print(f"  [Rollback] JANUS orientation comparison check failed, rolling back alpha={self.alpha:.4f} weights...")
                    for name in self.target_layers:
                        module = self.model.get_submodule(name)
                        delta_w = self.delta_w_dict[name].to(self.device)
                        P = self.P_old[name].to(self.device)
                        delta_w_projed = self.alpha * (delta_w - P)
                        module.weight.data.sub_(delta_w_projed)
                    self.alpha *= self.beta
                    
            if self.alpha == 1.0:
                print("Converged with a full step, terminating projection process.")
                shutil.rmtree(self.jac_dir_old, ignore_errors=True)
                shutil.rmtree(self.jac_dir_new, ignore_errors=True)
                break
            else:
                print(f"Converged with alpha={self.alpha:.4f}, updating old Jacobian cache for the next iteration...")
                shutil.rmtree(self.jac_dir_old, ignore_errors=True)
                os.rename(self.jac_dir_new, self.jac_dir_old)
                os.makedirs(self.jac_dir_new, exist_ok=True)

                for name in self.target_layers:
                    delta_w = self.delta_w_dict[name]
                    P = self.P_old[name]
                    self.delta_w_dict[name] = (1 - self.alpha) * delta_w + self.alpha * P
                self.Gram_old = self.Gram_new
                self.P_old = self.P_new
                
        return self.model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str)
    parser.add_argument("--finetuned_model_path", type=str)
    parser.add_argument("--finetuned_model_type", type=str, choices=["pissa", "lora", "ff"])
    parser.add_argument("--residual_model_path", type=str, default=None)
    parser.add_argument("--replay_dataset", type=str, default="nqopen")
    parser.add_argument("--replay_size", type=int, default=256)
    parser.add_argument("--replay_seed", type=int, default=233)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--r_jac_approx", type=int, default=32)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    print(f"Projected model will be saved to {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    
    tokenizer_config = json.load(open(os.path.join(args.finetuned_model_path, "tokenizer_config.json"), "r"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        model_max_length=tokenizer_config["model_max_length"],
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    knowledge_dataset = get_knowledge_data(
        name=args.replay_dataset, 
        tokenizer=tokenizer, 
        model_id=args.base_model_id, 
        nsamples=args.replay_size, 
        seed=args.replay_seed
    )
    data_collector = DataCollatorForSupervisedDataset(tokenizer)
    
    dataloader = DataLoader(
        knowledge_dataset,
        batch_size=16,
        collate_fn=data_collector, 
        drop_last=True,      
        shuffle=False
    )
    
    print("Loading original and finetuned models...")
    if args.finetuned_model_type == "pissa":
        pissa_residual_model = AutoModelForCausalLM.from_pretrained(
            args.residual_model_path,
            dtype=torch.float16,
        )
        finetuned_model = PeftModel.from_pretrained(
            pissa_residual_model,
            args.finetuned_model_path,
            dtype=torch.float32,
        )
        finetuned_model = finetuned_model.merge_and_unload()
    elif args.finetuned_model_type == "lora":
        origin_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            dtype=torch.float16,
        )
        finetuned_model = PeftModel.from_pretrained(
            origin_model,
            args.finetuned_model_path,
            dtype=torch.float32,
        )
        finetuned_model = finetuned_model.merge_and_unload()
    elif args.finetuned_model_type == "ff":
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            args.finetuned_model_path,
            dtype=torch.float16,  # For computational precision during rectification, it is required that a fully-fine-tuned model is trained, saved, and loaded in float16 precision.
        )

    origin_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        dtype=torch.float16,
    )

    # This is only suitable for LLaMA models with a standard architecture. For other architectures, please modify the target layer selection accordingly.
    target_layers = [name for name, module in origin_model.named_modules() if isinstance(module, (Linear)) and "lm_head" not in name]
    print(f"Target layers for projection: {target_layers}")

    print("Initializing MultiStepAdaptiveRectifier...")
    rectifier = MultiStepAdaptiveRectifier(
        original_model=origin_model,
        finetuned_model=finetuned_model,
        target_layers=target_layers,
        cache_dir=f"{args.save_path}/jac_cache",
        r_jac_approx=args.r_jac_approx,
        delta_threshold=args.threshold,
        beta=args.beta,
    )
    print("Starting multi-step adaptive rectification...")
    projected_model = rectifier.multi_step_adaptive_rectification(dataloader)
    
    projected_model = projected_model.to(torch.bfloat16)
    projected_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Projected model and tokenizer saved to {args.save_path}.")
