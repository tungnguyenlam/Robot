# Project: Learning dexterous robot manipulation via human-ego centric videos informed by graph geometry
## Month 1: Foundation and Environment Setup
- Literature & Math Review: Deep dive into the theoretical formulations of the core baselines: flow matching in StableHand [1], world-space interaction tracking in EgoGrasp [2], and world-grounded lifting in WHOLE [3].
- Data Pipeline Construction: Set up the training environments and download standard egocentric datasets (e.g., Ego4D, EPIC-Kitchens, HOI4D).
- Deliverable: A unified codebase skeleton and data loader capable of parsing egocentric video inputs into a standard format for 3D hand and object pose extraction.

### Tracking
- Read EgoGrasp
## Month 2: Baseline Reproduction and Bottleneck Profiling
- Execute Baselines: Run the pre-trained inference models for StableHand, EgoGrasp, and WHOLE to extract 3D hand-object trajectories.
- Failure Case Analysis: Systematically profile where these baselines fail in dexterous manipulation contexts (e.g., occlusion, penetration, temporal jitter).
- Deliverable: A quantitative and qualitative baseline evaluation report identifying the exact failure modes that explicit geometric modeling will solve.
## Month 3: Graph Representation and Geometric Formulation
- Define the Scene Graph: Formulate the hand-object interaction as a spatial-temporal graph. Define nodes (hand joints, object bounding boxes/keypoints) and edges (physical contact, spatial proximity, temporal continuity).
- Feature Extraction: Develop the GNN/Transformer layers needed to encode the egocentric visual features into this structural graph prior.
- Deliverable: A functional module that converts raw baseline outputs and visual features into continuous graph embeddings.
## Month 4: Graph-Guided Diffusion Model Development
- Modify the Diffusion Process: Inject the graph embeddings into the diffusion/flow-matching backbone. Instead of denoising independent points, denoise the graph structure to explicitly enforce geometric constraints [4,5] (preventing hand-object penetration).
- Loss Function Engineering: Design physical constraint losses (e.g., contact consistency, temporal smoothness) and integrate them into the training objective.
- Deliverable: A working prototype of the graph-conditioned diffusion model capable of generating refined, physically plausible hand-object trajectories.
## Month 5: VLA Integration and Policy Training
- Trajectory Retargeting: Map the refined human hand-object trajectories into robotic action spaces (e.g., joint angles, end-effector poses) suitable for your specific robotic embodiment.
- VLA Policy Distillation: Use the generated, high-quality interaction data to fine-tune a baseline Vision-Language-Action model.
- Deliverable: A simulated dexterous manipulation policy executing tasks derived from the egocentric video data.
## Month 6: Simulation & Real-World Validation and Policy Refinement
- Simulation Benchmarking: Deploy the trained VLA policy into a high-fidelity, GPU-accelerated physics framework like Isaac Lab or SAPIEN to rigorously test contact-rich dexterous manipulation tasks. Measure grasp success rates, contact stability, and kinematic accuracy against policies trained purely on raw baselines.
- Sim-to-Real Deployment: Transfer the policy to a physical robot setup using structured domain randomization and spatial feature augmentation. Evaluate the zero-shot or few-shot robustness of the graph-diffusion data against real-world sensory noise, visual domain shifts, and unmodeled physical dynamics.
- Failure Analysis & Iterative Refinement: Systematically analyze failure modes during physical and simulated rollouts, such as kinematic singularities, unstable force closures, or trajectory jitters, and fine-tune the graph constraints or VLA retargeting module to close the loop on these specific bottlenecks.
- Deliverable: A comprehensive empirical validation report and finalized codebase demonstrating a quantifiable performance leap in physical and simulated dexterous tasks.

## Citation
[1] StableHand: Quality-Aware Flow Matching for World-Space Dual-Hand Motion Estimation from
Egocentric Video, NeurIPS 2026
[2] Fu, Hongming, et al. "Egograsp: World-space hand-object interaction estimation from egocentric videos." arXiv preprint arXiv:2601.01050 (2026).
[3] Ye, Yufei, et al. "Whole: World-grounded hand-object lifted from egocentric videos." arXiv preprint arXiv:2602.22209 (2026).
[4] Cruciani, Silvia, et al. "Dexterous manipulation graphs." 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018.
[5] Vosylius, Vitalis, and Edward Johns. "Instant policy: In-context imitation learning via graph diffusion." International Conference on Learning Representations 2025.
