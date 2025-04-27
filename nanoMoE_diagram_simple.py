"""
nanoMoE Simple Diagram Generator

This script generates a simple Mermaid diagram code for visualizing the structure 
and information flow of the nanoMoE repository, focusing on the Mixture-of-Experts architecture.
"""

def generate_mermaid_diagram():
    """
    Generate a Mermaid diagram showing the MoE architecture and information flow.
    """
    mermaid_code = """
```mermaid
graph TD
    %% Main Model Structure
    Input[Input Tokens] --> Embeddings[Token + Position Embeddings]
    Embeddings --> TransformerBlocks[Transformer Blocks]
    TransformerBlocks --> FinalLN[Final LayerNorm]
    FinalLN --> OutputHead[Output Linear Layer]
    OutputHead --> Output[Output Logits/Predictions]
    
    %% Transformer Block Types
    TransformerBlocks --> RegularBlock[Regular Transformer Block]
    TransformerBlocks --> MoEBlock[MoE Transformer Block]
    
    %% Regular Block Structure
    RegularBlock --> |1. First Step| RegLN1[LayerNorm]
    RegLN1 --> RegAttn[Causal Self-Attention]
    RegAttn --> |+ Residual| RegResid1[Add & Norm]
    RegResid1 --> RegLN2[LayerNorm]
    RegLN2 --> RegFFN[Feed-Forward Network]
    RegFFN --> |+ Residual| RegResid2[Add & Norm]
    
    %% MoE Block Structure
    MoEBlock --> |1. First Step| MoELN1[LayerNorm]
    MoELN1 --> MoEAttn[Causal Self-Attention]
    MoEAttn --> |+ Residual| MoEResid1[Add & Norm]
    MoEResid1 --> MoELN2[LayerNorm]
    MoELN2 --> MoELayer[MoE Layer]
    MoELayer --> |+ Residual| MoEResid2[Add & Norm]
    
    %% MoE Layer Components
    MoELayer --> Router[Router]
    MoELayer --> Experts[MLP Experts]
    
    %% Router Process
    Router --> RouterLogits[Compute Logits]
    RouterLogits --> |Optional| AddNoise[Add Noise]
    AddNoise --> TopK[Select Top-K Experts]
    TopK --> RouterProbs[Compute Expert Probabilities]
    
    %% Expert Process
    RouterProbs --> ExpertCapacity[Check Expert Capacity]
    ExpertCapacity --> DropTokens[Drop Tokens if Needed]
    DropTokens --> ExpertBatches[Create Expert Batches]
    ExpertBatches --> ExpertForward[Forward Through Experts]
    ExpertForward --> WeightedSum[Weighted Sum of Expert Outputs]
    
    %% Auxiliary Losses
    Router --> LoadBalancingLoss[Load Balancing Loss]
    Router --> RouterZLoss[Router Z-Loss]
    LoadBalancingLoss --> AuxLosses[Auxiliary Losses]
    RouterZLoss --> AuxLosses
    
    %% Styling
    classDef main fill:#f9f,stroke:#333,stroke-width:2px;
    classDef block fill:#bbf,stroke:#33f,stroke-width:1px;
    classDef moe fill:#bfb,stroke:#3f3,stroke-width:1px;
    classDef loss fill:#fbb,stroke:#f33,stroke-width:1px;
    classDef io fill:#ffffcc,stroke:#999,stroke-width:1px;
    
    class Input,Output,Embeddings io;
    class TransformerBlocks,RegularBlock,MoEBlock main;
    class RegLN1,RegAttn,RegResid1,RegLN2,RegFFN,RegResid2 block;
    class MoELN1,MoEAttn,MoEResid1,MoELN2,MoELayer,MoEResid2 block;
    class Router,Experts,RouterLogits,AddNoise,TopK,RouterProbs,ExpertCapacity,DropTokens,ExpertBatches,ExpertForward,WeightedSum moe;
    class LoadBalancingLoss,RouterZLoss,AuxLosses loss;
```
"""
    return mermaid_code

def generate_class_diagram():
    """
    Generate a class diagram showing the key classes in nanoMoE.
    """
    class_diagram = """
```mermaid
classDiagram
    %% Main Model Classes
    class GPT {
        +ModuleDict transformer
        +forward(idx, targets)
    }
    
    class Block {
        +LayerNorm ln_1
        +CausalSelfAttention attn
        +LayerNorm ln_2
        +MLP ffnn
        +forward(x)
    }
    
    class MoEBlock {
        +LayerNorm ln_1
        +CausalSelfAttention attn
        +LayerNorm ln_2
        +MOELayer mlp
        +forward(x)
    }
    
    %% MoE Components
    class MOELayer {
        +Router router
        +MLPExperts experts
        +forward(x)
    }
    
    class MLPExperts {
        +Parameter c_fc
        +Parameter c_proj
        +Parameter fc_bias
        +Parameter proj_bias
        +GELU gelu
        +Dropout dropout
        +forward(x)
    }
    
    class BasicSoftmaxRouter {
        +int top_k
        +bool use_noisy_top_k
        +Linear w_g
        +Linear w_noise
        +forward(x)
    }
    
    class Router {
        +int top_k
        +bool use_noisy_top_k
        +float capacity_factor
        +Linear w_g
        +Linear w_noise
        +forward(x)
    }
    
    %% Standard Components
    class MLP {
        +Linear c_fc
        +GELU gelu
        +Linear c_proj
        +Dropout dropout
        +forward(x)
    }
    
    class CausalSelfAttention {
        +Linear c_attn
        +Linear c_proj
        +Dropout attn_dropout
        +Dropout resid_dropout
        +int H
        +int d
        +Buffer mask
        +forward(x)
    }
    
    %% Relationships
    GPT --> Block: contains
    GPT --> MoEBlock: contains
    Block --> CausalSelfAttention: uses
    Block --> MLP: uses
    MoEBlock --> CausalSelfAttention: uses
    MoEBlock --> MOELayer: uses
    MOELayer --> Router: uses
    MOELayer --> MLPExperts: uses
    Router --|> BasicSoftmaxRouter: extends
```
"""
    return class_diagram

def main():
    """
    Main function to generate diagrams.
    """
    print("Generating nanoMoE simple diagrams...\n")
    
    print("1. Mermaid Architecture Diagram:")
    print(generate_mermaid_diagram())
    print("\n" + "-"*80 + "\n")
    
    print("2. Mermaid Class Diagram:")
    print(generate_class_diagram())
    
    print("\nDiagrams generated successfully. Copy the code blocks into a Mermaid diagram editor.")
    print("For Mermaid diagrams: https://mermaid.live/")

if __name__ == "__main__":
    main()