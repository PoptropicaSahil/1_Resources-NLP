"""
nanoMoE Diagram Generator

This script generates diagram code for visualizing the structure and information flow
of the nanoMoE repository. The output can be used with standard diagramming tools
like Mermaid, PlantUML, or Graphviz.
"""

def generate_mermaid_diagram():
    """
    Generate a Mermaid diagram showing the structure and information flow of nanoMoE.
    """
    mermaid_code = """
```mermaid
graph TD
    %% Main Model Components
    GPT[GPT Model] --> Blocks[Transformer Blocks]
    Blocks --> |Every P-th layer| MoEBlock[MoE Block]
    Blocks --> |Other layers| Block[Regular Block]
    
    %% Regular Block Components
    Block --> LayerNorm1[LayerNorm]
    LayerNorm1 --> SelfAttention[Causal Self Attention]
    Block --> LayerNorm2[LayerNorm]
    LayerNorm2 --> FFNN[Feed Forward Neural Network]
    
    %% MoE Block Components
    MoEBlock --> MoELayerNorm1[LayerNorm]
    MoELayerNorm1 --> MoESelfAttention[Causal Self Attention]
    MoEBlock --> MoELayerNorm2[LayerNorm]
    MoELayerNorm2 --> MoELayer[MoE Layer]
    
    %% MoE Layer Components
    MoELayer --> Router[Router]
    MoELayer --> Experts[MLP Experts]
    
    %% Router Components
    Router --> BasicRouter[Basic Softmax Router]
    Router --> FullRouter[Full Softmax Router]
    
    %% Auxiliary Losses
    AuxLosses[Auxiliary Losses] --> LoadBalancingLoss[Load Balancing Loss]
    AuxLosses --> RouterZLoss[Router Z-Loss]
    
    %% Information Flow
    Input[Input Tokens] --> |Embedding| GPT
    GPT --> |Forward Pass| Output[Output Logits]
    
    %% Router Information Flow
    Router --> |Top-K Selection| TopK[Top-K Experts]
    TopK --> |Expert Weights| Experts
    Experts --> |Weighted Sum| MoEOutput[MoE Output]
    
    %% Styling
    classDef main fill:#f9f,stroke:#333,stroke-width:2px;
    classDef component fill:#bbf,stroke:#33f,stroke-width:1px;
    classDef router fill:#bfb,stroke:#3f3,stroke-width:1px;
    classDef loss fill:#fbb,stroke:#f33,stroke-width:1px;
    
    class GPT,MoEBlock,Block main;
    class LayerNorm1,LayerNorm2,SelfAttention,FFNN,MoELayer,Experts component;
    class Router,BasicRouter,FullRouter router;
    class AuxLosses,LoadBalancingLoss,RouterZLoss loss;
```
"""
    return mermaid_code

def generate_plantuml_diagram():
    """
    Generate a PlantUML diagram showing the structure and information flow of nanoMoE.
    """
    plantuml_code = """
```plantuml
@startuml NanoMoE Architecture

package "nanoMoE" {
    [GPT Model] as GPT
    
    package "Transformer Blocks" {
        [Regular Block] as Block
        [MoE Block] as MoEBlock
        note right of MoEBlock: Every P-th layer
    }
    
    package "Regular Block Components" {
        [LayerNorm] as LN1
        [Causal Self Attention] as CSA
        [LayerNorm] as LN2
        [Feed Forward Neural Network] as FFNN
    }
    
    package "MoE Block Components" {
        [LayerNorm] as MoELN1
        [Causal Self Attention] as MoECSA
        [LayerNorm] as MoELN2
        [MoE Layer] as MoELayer
    }
    
    package "MoE Layer Components" {
        [Router] as Router
        [MLP Experts] as Experts
    }
    
    package "Router Types" {
        [Basic Softmax Router] as BasicRouter
        [Full Softmax Router] as FullRouter
    }
    
    package "Auxiliary Losses" {
        [Load Balancing Loss] as LBLoss
        [Router Z-Loss] as ZLoss
    }
    
    ' Connections
    GPT --> Block
    GPT --> MoEBlock
    
    Block --> LN1
    LN1 --> CSA
    Block --> LN2
    LN2 --> FFNN
    
    MoEBlock --> MoELN1
    MoELN1 --> MoECSA
    MoEBlock --> MoELN2
    MoELN2 --> MoELayer
    
    MoELayer --> Router
    MoELayer --> Experts
    
    Router --> BasicRouter
    Router --> FullRouter
    
    Router ..> LBLoss: contributes to
    Router ..> ZLoss: contributes to
    
    ' Information Flow
    [Input Tokens] as Input
    [Output Logits] as Output
    
    Input --> GPT: embedding
    GPT --> Output: forward pass
    
    Router --> Experts: top-k selection
    Experts --> MoELayer: weighted sum
}

@enduml
```
"""
    return plantuml_code

def generate_graphviz_diagram():
    """
    Generate a Graphviz DOT diagram showing the structure and information flow of nanoMoE.
    """
    graphviz_code = """
```dot
digraph NanoMoE {
    // Graph settings
    rankdir=TB;
    node [shape=box, style=filled, fontname="Arial"];
    edge [fontname="Arial", fontsize=10];
    
    // Main components
    GPT [label="GPT Model", fillcolor="#ffccff"];
    Block [label="Regular Block", fillcolor="#ffccff"];
    MoEBlock [label="MoE Block", fillcolor="#ffccff"];
    
    // Regular Block components
    LN1 [label="LayerNorm", fillcolor="#ccccff"];
    CSA [label="Causal Self Attention", fillcolor="#ccccff"];
    LN2 [label="LayerNorm", fillcolor="#ccccff"];
    FFNN [label="Feed Forward Neural Network", fillcolor="#ccccff"];
    
    // MoE Block components
    MoELN1 [label="LayerNorm", fillcolor="#ccccff"];
    MoECSA [label="Causal Self Attention", fillcolor="#ccccff"];
    MoELN2 [label="LayerNorm", fillcolor="#ccccff"];
    MoELayer [label="MoE Layer", fillcolor="#ccccff"];
    
    // MoE Layer components
    Router [label="Router", fillcolor="#ccffcc"];
    Experts [label="MLP Experts", fillcolor="#ccccff"];
    
    // Router types
    BasicRouter [label="Basic Softmax Router", fillcolor="#ccffcc"];
    FullRouter [label="Full Softmax Router", fillcolor="#ccffcc"];
    
    // Auxiliary Losses
    LBLoss [label="Load Balancing Loss", fillcolor="#ffcccc"];
    ZLoss [label="Router Z-Loss", fillcolor="#ffcccc"];
    
    // Input/Output
    Input [label="Input Tokens", shape=ellipse, fillcolor="#ffffcc"];
    Output [label="Output Logits", shape=ellipse, fillcolor="#ffffcc"];
    
    // Connections
    GPT -> {Block, MoEBlock} [label="contains"];
    
    Block -> LN1 -> CSA;
    Block -> LN2 -> FFNN;
    
    MoEBlock -> MoELN1 -> MoECSA;
    MoEBlock -> MoELN2 -> MoELayer;
    
    MoELayer -> Router;
    MoELayer -> Experts;
    
    Router -> {BasicRouter, FullRouter} [label="types"];
    
    Router -> LBLoss [style=dashed, label="contributes to"];
    Router -> ZLoss [style=dashed, label="contributes to"];
    
    // Information Flow
    Input -> GPT [label="embedding"];
    GPT -> Output [label="forward pass"];
    
    // Expert Selection Flow
    Router -> Experts [label="top-k selection"];
    Experts -> MoELayer [label="weighted sum"];
    
    // Subgraphs for organization
    subgraph cluster_blocks {
        label="Transformer Blocks";
        Block; MoEBlock;
    }
    
    subgraph cluster_regular_block {
        label="Regular Block Components";
        LN1; CSA; LN2; FFNN;
    }
    
    subgraph cluster_moe_block {
        label="MoE Block Components";
        MoELN1; MoECSA; MoELN2; MoELayer;
    }
    
    subgraph cluster_moe_layer {
        label="MoE Layer Components";
        Router; Experts;
    }
    
    subgraph cluster_router_types {
        label="Router Types";
        BasicRouter; FullRouter;
    }
    
    subgraph cluster_losses {
        label="Auxiliary Losses";
        LBLoss; ZLoss;
    }
}
```
"""
    return graphviz_code

def generate_class_diagram():
    """
    Generate a class diagram showing the relationships between classes in nanoMoE.
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
    
    %% Auxiliary Losses
    class LoadBalancingLoss {
        +compute_loss(indices, expert_probs)
    }
    
    class RouterZLoss {
        +compute_loss(router_logits)
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
    Router --> LoadBalancingLoss: contributes to
    Router --> RouterZLoss: contributes to
```
"""
    return class_diagram

def generate_information_flow_diagram():
    """
    Generate a diagram specifically showing the information flow in nanoMoE.
    """
    flow_diagram = """
```mermaid
flowchart TD
    %% Input Processing
    Input[Input Tokens] --> Embedding[Token + Position Embeddings]
    Embedding --> Dropout
    
    %% Transformer Blocks
    Dropout --> BlocksLoop{For each block}
    BlocksLoop --> |Regular Block| RegularBlock[Regular Block Processing]
    BlocksLoop --> |MoE Block| MoEBlock[MoE Block Processing]
    
    %% Regular Block Processing
    RegularBlock --> RegLN1[LayerNorm]
    RegLN1 --> RegAttn[Self-Attention]
    RegAttn --> RegResid1[Residual Connection]
    RegResid1 --> RegLN2[LayerNorm]
    RegLN2 --> RegFFNN[Feed-Forward Network]
    RegFFNN --> RegResid2[Residual Connection]
    
    %% MoE Block Processing
    MoEBlock --> MoELN1[LayerNorm]
    MoELN1 --> MoEAttn[Self-Attention]
    MoEAttn --> MoEResid1[Residual Connection]
    MoEResid1 --> MoELN2[LayerNorm]
    MoELN2 --> RouterProcess[Router Processing]
    
    %% Router Processing
    RouterProcess --> RouterLogits[Compute Router Logits]
    RouterLogits --> |Optional| AddNoise[Add Noise]
    AddNoise --> TopK[Select Top-K Experts]
    TopK --> RouterProbs[Compute Expert Probabilities]
    RouterProbs --> ExpertCapacity[Compute Expert Capacity]
    ExpertCapacity --> DropTokens[Drop Tokens if Capacity Exceeded]
    DropTokens --> ExpertBatches[Create Expert Batches]
    
    %% Expert Processing
    ExpertBatches --> ExpertForward[Forward Pass Through Experts]
    ExpertForward --> WeightedSum[Weighted Sum of Expert Outputs]
    WeightedSum --> MoEResid2[Residual Connection]
    
    %% Final Processing
    RegResid2 --> NextBlock[Next Block]
    MoEResid2 --> NextBlock
    NextBlock --> |Last Block| FinalLN[Final LayerNorm]
    FinalLN --> OutputHead[Output Linear Layer]
    OutputHead --> |Training| LossComputation[Loss Computation]
    OutputHead --> |Inference| Output[Output Logits]
    
    %% Loss Computation
    LossComputation --> MainLoss[Cross-Entropy Loss]
    LossComputation --> |If MoE| AuxLosses[Auxiliary Losses]
    AuxLosses --> LoadBalancingLoss[Load Balancing Loss]
    AuxLosses --> RouterZLoss[Router Z-Loss]
    MainLoss --> CombinedLoss[Combined Loss]
    LoadBalancingLoss --> CombinedLoss
    RouterZLoss --> CombinedLoss
    
    %% Styling
    classDef process fill:#f9f,stroke:#333,stroke-width:1px;
    classDef router fill:#bfb,stroke:#3f3,stroke-width:1px;
    classDef loss fill:#fbb,stroke:#f33,stroke-width:1px;
    classDef io fill:#bbf,stroke:#33f,stroke-width:1px;
    
    class Input,Output,Embedding io;
    class RouterProcess,RouterLogits,AddNoise,TopK,RouterProbs,ExpertCapacity,DropTokens,ExpertBatches router;
    class MainLoss,AuxLosses,LoadBalancingLoss,RouterZLoss,CombinedLoss loss;
    class RegularBlock,MoEBlock,ExpertForward,WeightedSum process;
```
"""
    return flow_diagram

def main():
    """
    Main function to generate all diagram types.
    """
    print("Generating nanoMoE diagrams...\n")
    
    print("1. Mermaid Structure Diagram:")
    print(generate_mermaid_diagram())
    print("\n" + "-"*80 + "\n")
    
    print("2. PlantUML Structure Diagram:")
    print(generate_plantuml_diagram())
    print("\n" + "-"*80 + "\n")
    
    print("3. Graphviz DOT Diagram:")
    print(generate_graphviz_diagram())
    print("\n" + "-"*80 + "\n")
    
    print("4. Mermaid Class Diagram:")
    print(generate_class_diagram())
    print("\n" + "-"*80 + "\n")
    
    print("5. Mermaid Information Flow Diagram:")
    print(generate_information_flow_diagram())
    
    print("\nDiagrams generated successfully. Copy the code blocks into your preferred diagramming tool.")
    print("For Mermaid diagrams: https://mermaid.live/")
    print("For PlantUML diagrams: https://www.planttext.com/")
    print("For Graphviz diagrams: https://dreampuf.github.io/GraphvizOnline/")

if __name__ == "__main__":
    main()