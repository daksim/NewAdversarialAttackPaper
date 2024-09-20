# Latest Adversarial Attack Papers
**update at 2024-09-20 16:14:57**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Defending against Reverse Preference Attacks is Difficult**

cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12914v1) [paper-pdf](http://arxiv.org/pdf/2409.12914v1)

**Authors**: Domenic Rosati, Giles Edkins, Harsh Raj, David Atanasov, Subhabrata Majumdar, Janarthanan Rajendran, Frank Rudzicz, Hassan Sajjad

**Abstract**: While there has been progress towards aligning Large Language Models (LLMs) with human values and ensuring safe behaviour at inference time, safety-aligned LLMs are known to be vulnerable to training-time attacks such as supervised fine-tuning (SFT) on harmful datasets. In this paper, we ask if LLMs are vulnerable to adversarial reinforcement learning. Motivated by this goal, we propose Reverse Preference Attacks (RPA), a class of attacks to make LLMs learn harmful behavior using adversarial reward during reinforcement learning from human feedback (RLHF). RPAs expose a critical safety gap of safety-aligned LLMs in RL settings: they easily explore the harmful text generation policies to optimize adversarial reward. To protect against RPAs, we explore a host of mitigation strategies. Leveraging Constrained Markov-Decision Processes, we adapt a number of mechanisms to defend against harmful fine-tuning attacks into the RL setting. Our experiments show that ``online" defenses that are based on the idea of minimizing the negative log likelihood of refusals -- with the defender having control of the loss function -- can effectively protect LLMs against RPAs. However, trying to defend model weights using ``offline" defenses that operate under the assumption that the defender has no control over the loss function are less effective in the face of RPAs. These findings show that attacks done using RL can be used to successfully undo safety alignment in open-weight LLMs and use them for malicious purposes.



## **2. Boosting Certified Robustness for Time Series Classification with Efficient Self-Ensemble**

cs.LG

6 figures, 4 tables, 10 pages

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.02802v3) [paper-pdf](http://arxiv.org/pdf/2409.02802v3)

**Authors**: Chang Dong, Zhengyang Li, Liangwei Zheng, Weitong Chen, Wei Emma Zhang

**Abstract**: Recently, the issue of adversarial robustness in the time series domain has garnered significant attention. However, the available defense mechanisms remain limited, with adversarial training being the predominant approach, though it does not provide theoretical guarantees. Randomized Smoothing has emerged as a standout method due to its ability to certify a provable lower bound on robustness radius under $\ell_p$-ball attacks. Recognizing its success, research in the time series domain has started focusing on these aspects. However, existing research predominantly focuses on time series forecasting, or under the non-$\ell_p$ robustness in statistic feature augmentation for time series classification~(TSC). Our review found that Randomized Smoothing performs modestly in TSC, struggling to provide effective assurances on datasets with poor robustness. Therefore, we propose a self-ensemble method to enhance the lower bound of the probability confidence of predicted labels by reducing the variance of classification margins, thereby certifying a larger radius. This approach also addresses the computational overhead issue of Deep Ensemble~(DE) while remaining competitive and, in some cases, outperforming it in terms of robustness. Both theoretical analysis and experimental results validate the effectiveness of our method, demonstrating superior performance in robustness testing compared to baseline approaches.



## **3. Deep generative models as an adversarial attack strategy for tabular machine learning**

cs.LG

Accepted at ICMLC 2024 (International Conference on Machine Learning  and Cybernetics)

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12642v1) [paper-pdf](http://arxiv.org/pdf/2409.12642v1)

**Authors**: Salijona Dyrmishi, Mihaela Cătălina Stoian, Eleonora Giunchiglia, Maxime Cordy

**Abstract**: Deep Generative Models (DGMs) have found application in computer vision for generating adversarial examples to test the robustness of machine learning (ML) systems. Extending these adversarial techniques to tabular ML presents unique challenges due to the distinct nature of tabular data and the necessity to preserve domain constraints in adversarial examples. In this paper, we adapt four popular tabular DGMs into adversarial DGMs (AdvDGMs) and evaluate their effectiveness in generating realistic adversarial examples that conform to domain constraints.



## **4. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2402.18607v3) [paper-pdf](http://arxiv.org/pdf/2402.18607v3)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.



## **5. Adversarial Attack for Explanation Robustness of Rationalization Models**

cs.CL

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2408.10795v3) [paper-pdf](http://arxiv.org/pdf/2408.10795v3)

**Authors**: Yuankai Zhang, Lingxiao Kong, Haozhao Wang, Ruixuan Li, Jun Wang, Yuhua Li, Wei Liu

**Abstract**: Rationalization models, which select a subset of input text as rationale-crucial for humans to understand and trust predictions-have recently emerged as a prominent research area in eXplainable Artificial Intelligence. However, most of previous studies mainly focus on improving the quality of the rationale, ignoring its robustness to malicious attack. Specifically, whether the rationalization models can still generate high-quality rationale under the adversarial attack remains unknown. To explore this, this paper proposes UAT2E, which aims to undermine the explainability of rationalization models without altering their predictions, thereby eliciting distrust in these models from human users. UAT2E employs the gradient-based search on triggers and then inserts them into the original input to conduct both the non-target and target attack. Experimental results on five datasets reveal the vulnerability of rationalization models in terms of explanation, where they tend to select more meaningless tokens under attacks. Based on this, we make a series of recommendations for improving rationalization models in terms of explanation.



## **6. Designing an attack-defense game: how to increase robustness of financial transaction models via a competition**

cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2308.11406v3) [paper-pdf](http://arxiv.org/pdf/2308.11406v3)

**Authors**: Alexey Zaytsev, Maria Kovaleva, Alex Natekin, Evgeni Vorsin, Valerii Smirnov, Georgii Smirnov, Oleg Sidorshin, Alexander Senin, Alexander Dudin, Dmitry Berestnev

**Abstract**: Banks routinely use neural networks to make decisions. While these models offer higher accuracy, they are susceptible to adversarial attacks, a risk often overlooked in the context of event sequences, particularly sequences of financial transactions, as most works consider computer vision and NLP modalities.   We propose a thorough approach to studying these risks: a novel type of competition that allows a realistic and detailed investigation of problems in financial transaction data. The participants directly oppose each other, proposing attacks and defenses -- so they are examined in close-to-real-life conditions.   The paper outlines our unique competition structure with direct opposition of participants, presents results for several different top submissions, and analyzes the competition results. We also introduce a new open dataset featuring financial transactions with credit default labels, enhancing the scope for practical research and development.



## **7. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

cs.CR

Accepted at NDSS 2025

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2311.00207v2) [paper-pdf](http://arxiv.org/pdf/2311.00207v2)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer protocols, and wireless domain constraints. This paper proposes Magmaw, a novel wireless attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on downstream applications. We adopt the widely-used defenses to verify the resilience of Magmaw. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of strong defense mechanisms. Furthermore, we validate the performance of Magmaw in two case studies: encrypted communication channel and channel modality-based ML model.



## **8. TEAM: Temporal Adversarial Examples Attack Model against Network Intrusion Detection System Applied to RNN**

cs.CR

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12472v1) [paper-pdf](http://arxiv.org/pdf/2409.12472v1)

**Authors**: Ziyi Liu, Dengpan Ye, Long Tang, Yunming Zhang, Jiacheng Deng

**Abstract**: With the development of artificial intelligence, neural networks play a key role in network intrusion detection systems (NIDS). Despite the tremendous advantages, neural networks are susceptible to adversarial attacks. To improve the reliability of NIDS, many research has been conducted and plenty of solutions have been proposed. However, the existing solutions rarely consider the adversarial attacks against recurrent neural networks (RNN) with time steps, which would greatly affect the application of NIDS in real world. Therefore, we first propose a novel RNN adversarial attack model based on feature reconstruction called \textbf{T}emporal adversarial \textbf{E}xamples \textbf{A}ttack \textbf{M}odel \textbf{(TEAM)}, which applied to time series data and reveals the potential connection between adversarial and time steps in RNN. That is, the past adversarial examples within the same time steps can trigger further attacks on current or future original examples. Moreover, TEAM leverages Time Dilation (TD) to effectively mitigates the effect of temporal among adversarial examples within the same time steps. Experimental results show that in most attack categories, TEAM improves the misjudgment rate of NIDS on both black and white boxes, making the misjudgment rate reach more than 96.68%. Meanwhile, the maximum increase in the misjudgment rate of the NIDS for subsequent original samples exceeds 95.57%.



## **9. Object-fabrication Targeted Attack for Object Detection**

cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2212.06431v3) [paper-pdf](http://arxiv.org/pdf/2212.06431v3)

**Authors**: Xuchong Zhang, Changfeng Sun, Haoliang Han, Hongbin Sun

**Abstract**: Recent studies have demonstrated that object detection networks are usually vulnerable to adversarial examples. Generally, adversarial attacks for object detection can be categorized into targeted and untargeted attacks. Compared with untargeted attacks, targeted attacks present greater challenges and all existing targeted attack methods launch the attack by misleading detectors to mislabel the detected object as a specific wrong label. However, since these methods must depend on the presence of the detected objects within the victim image, they suffer from limitations in attack scenarios and attack success rates. In this paper, we propose a targeted feature space attack method that can mislead detectors to `fabricate' extra designated objects regardless of whether the victim image contains objects or not. Specifically, we introduce a guided image to extract coarse-grained features of the target objects and design an innovative dual attention mechanism to filter out the critical features of the target objects efficiently. The attack performance of the proposed method is evaluated on MS COCO and BDD100K datasets with FasterRCNN and YOLOv5. Evaluation results indicate that the proposed targeted feature space attack method shows significant improvements in terms of image-specific, universality, and generalization attack performance, compared with the previous targeted attack for object detection.



## **10. Towards Physically-Realizable Adversarial Attacks in Embodied Vision Navigation**

cs.CV

8 pages, 6 figures, submitted to the 2025 IEEE International  Conference on Robotics & Automation (ICRA)

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.10071v2) [paper-pdf](http://arxiv.org/pdf/2409.10071v2)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The deployment of embodied navigation agents in safety-critical environments raises concerns about their vulnerability to adversarial attacks on deep neural networks. However, current attack methods often lack practicality due to challenges in transitioning from the digital to the physical world, while existing physical attacks for object detection fail to achieve both multi-view effectiveness and naturalness. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches with learnable textures and opacity to objects. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which uses feedback from the navigation model to optimize the patch's texture. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, where opacity is refined after texture optimization. Experimental results show our adversarial patches reduce navigation success rates by about 40%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: [https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].



## **11. Typography Leads Semantic Diversifying: Amplifying Adversarial Transferability across Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2405.20090v2) [paper-pdf](http://arxiv.org/pdf/2405.20090v2)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jiahang Cao, Le Yang, Jize Zhang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Following the advent of the Artificial Intelligence (AI) era of large models, Multimodal Large Language Models (MLLMs) with the ability to understand cross-modal interactions between vision and text have attracted wide attention. Adversarial examples with human-imperceptible perturbation are shown to possess a characteristic known as transferability, which means that a perturbation generated by one model could also mislead another different model. Augmenting the diversity in input data is one of the most significant methods for enhancing adversarial transferability. This method has been certified as a way to significantly enlarge the threat impact under black-box conditions. Research works also demonstrate that MLLMs can be exploited to generate adversarial examples in the white-box scenario. However, the adversarial transferability of such perturbations is quite limited, failing to achieve effective black-box attacks across different models. In this paper, we propose the Typographic-based Semantic Transfer Attack (TSTA), which is inspired by: (1) MLLMs tend to process semantic-level information; (2) Typographic Attack could effectively distract the visual information captured by MLLMs. In the scenarios of Harmful Word Insertion and Important Information Protection, our TSTA demonstrates superior performance.



## **12. ITPatch: An Invisible and Triggered Physical Adversarial Patch against Traffic Sign Recognition**

cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12394v1) [paper-pdf](http://arxiv.org/pdf/2409.12394v1)

**Authors**: Shuai Yuan, Hongwei Li, Xingshuo Han, Guowen Xu, Wenbo Jiang, Tao Ni, Qingchuan Zhao, Yuguang Fang

**Abstract**: Physical adversarial patches have emerged as a key adversarial attack to cause misclassification of traffic sign recognition (TSR) systems in the real world. However, existing adversarial patches have poor stealthiness and attack all vehicles indiscriminately once deployed. In this paper, we introduce an invisible and triggered physical adversarial patch (ITPatch) with a novel attack vector, i.e., fluorescent ink, to advance the state-of-the-art. It applies carefully designed fluorescent perturbations to a target sign, an attacker can later trigger a fluorescent effect using invisible ultraviolet light, causing the TSR system to misclassify the sign and potentially resulting in traffic accidents. We conducted a comprehensive evaluation to investigate the effectiveness of ITPatch, which shows a success rate of 98.31% in low-light conditions. Furthermore, our attack successfully bypasses five popular defenses and achieves a success rate of 96.72%.



## **13. Enhancing 3D Robotic Vision Robustness by Minimizing Adversarial Mutual Information through a Curriculum Training Approach**

cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12379v1) [paper-pdf](http://arxiv.org/pdf/2409.12379v1)

**Authors**: Nastaran Darabi, Dinithi Jayasuriya, Devashri Naik, Theja Tulabandhula, Amit Ranjan Trivedi

**Abstract**: Adversarial attacks exploit vulnerabilities in a model's decision boundaries through small, carefully crafted perturbations that lead to significant mispredictions. In 3D vision, the high dimensionality and sparsity of data greatly expand the attack surface, making 3D vision particularly vulnerable for safety-critical robotics. To enhance 3D vision's adversarial robustness, we propose a training objective that simultaneously minimizes prediction loss and mutual information (MI) under adversarial perturbations to contain the upper bound of misprediction errors. This approach simplifies handling adversarial examples compared to conventional methods, which require explicit searching and training on adversarial samples. However, minimizing prediction loss conflicts with minimizing MI, leading to reduced robustness and catastrophic forgetting. To address this, we integrate curriculum advisors in the training setup that gradually introduce adversarial objectives to balance training and prevent models from being overwhelmed by difficult cases early in the process. The advisors also enhance robustness by encouraging training on diverse MI examples through entropy regularizers. We evaluated our method on ModelNet40 and KITTI using PointNet, DGCNN, SECOND, and PointTransformers, achieving 2-5% accuracy gains on ModelNet40 and a 5-10% mAP improvement in object detection. Our code is publicly available at https://github.com/nstrndrbi/Mine-N-Learn.



## **14. AirGapAgent: Protecting Privacy-Conscious Conversational Agents**

cs.CR

at CCS'24

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2405.05175v2) [paper-pdf](http://arxiv.org/pdf/2405.05175v2)

**Authors**: Eugene Bagdasarian, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.



## **15. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

cs.CR

21 pages, 7 figures

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2401.07261v4) [paper-pdf](http://arxiv.org/pdf/2401.07261v4)

**Authors**: Shoupeng Ren, Lipeng He, Tianyu Tu, Di Wu, Jian Liu, Kui Ren, Chun Chen

**Abstract**: Decentralized Finance (DeFi) incidents stemming from the exploitation of smart contract vulnerabilities have culminated in financial damages exceeding 3 billion US dollars. Existing defense mechanisms typically focus on detecting and reacting to malicious transactions executed by attackers that target victim contracts. However, with the emergence of private transaction pools where transactions are sent directly to miners without first appearing in public mempools, current detection tools face significant challenges in identifying attack activities effectively.   Based on the fact that most attack logic rely on deploying one or more intermediate smart contracts as supporting components to the exploitation of victim contracts, in this paper, we propose a new direction for detecting DeFi attacks that focuses on identifying adversarial contracts instead of adversarial transactions. Our approach allows us to leverage common attack patterns, code semantics and intrinsic characteristics found in malicious smart contracts to build the LookAhead system based on Machine Learning (ML) classifiers and a transformer model that is able to effectively distinguish adversarial contracts from benign ones, and make just-in-time predictions of potential zero-day attacks. Our contributions are three-fold: First, we construct a comprehensive dataset consisting of features extracted and constructed from recent contracts deployed on the Ethereum and BSC blockchains. Secondly, we design a condensed representation of smart contract programs called Pruned Semantic-Control Flow Tokenization (PSCFT) and use it to train a combination of ML models that understand the behaviour of malicious codes based on function calls, control flows and other pattern-conforming features. Lastly, we provide the complete implementation of LookAhead and the evaluation of its performance metrics for detecting adversarial contracts.



## **16. Bi-objective trail-planning for a robot team orienteering in a hazardous environment**

cs.RO

v0.0

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.12114v1) [paper-pdf](http://arxiv.org/pdf/2409.12114v1)

**Authors**: Cory M. Simon, Jeffrey Richley, Lucas Overbey, Darleen Perez-Lavin

**Abstract**: Teams of mobile [aerial, ground, or aquatic] robots have applications in resource delivery, patrolling, information-gathering, agriculture, forest fire fighting, chemical plume source localization and mapping, and search-and-rescue. Robot teams traversing hazardous environments -- with e.g. rough terrain or seas, strong winds, or adversaries capable of attacking or capturing robots -- should plan and coordinate their trails in consideration of risks of disablement, destruction, or capture. Specifically, the robots should take the safest trails, coordinate their trails to cooperatively achieve the team-level objective with robustness to robot failures, and balance the reward from visiting locations against risks of robot losses. Herein, we consider bi-objective trail-planning for a mobile team of robots orienteering in a hazardous environment. The hazardous environment is abstracted as a directed graph whose arcs, when traversed by a robot, present known probabilities of survival. Each node of the graph offers a reward to the team if visited by a robot (which e.g. delivers a good to or images the node). We wish to search for the Pareto-optimal robot-team trail plans that maximize two [conflicting] team objectives: the expected (i) team reward and (ii) number of robots that survive the mission. A human decision-maker can then select trail plans that balance, according to their values, reward and robot survival. We implement ant colony optimization, guided by heuristics, to search for the Pareto-optimal set of robot team trail plans. As a case study, we illustrate with an information-gathering mission in an art museum.



## **17. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2407.20361v2) [paper-pdf](http://arxiv.org/pdf/2407.20361v2)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.



## **18. Adversarial attacks on neural networks through canonical Riemannian foliations**

stat.ML

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2203.00922v3) [paper-pdf](http://arxiv.org/pdf/2203.00922v3)

**Authors**: Eliot Tron, Nicolas Couellan, Stéphane Puechmorel

**Abstract**: Deep learning models are known to be vulnerable to adversarial attacks. Adversarial learning is therefore becoming a crucial task. We propose a new vision on neural network robustness using Riemannian geometry and foliation theory. The idea is illustrated by creating a new adversarial attack that takes into account the curvature of the data space. This new adversarial attack, called the two-step spectral attack is a piece-wise linear approximation of a geodesic in the data space. The data space is treated as a (degenerate) Riemannian manifold equipped with the pullback of the Fisher Information Metric (FIM) of the neural network. In most cases, this metric is only semi-definite and its kernel becomes a central object to study. A canonical foliation is derived from this kernel. The curvature of transverse leaves gives the appropriate correction to get a two-step approximation of the geodesic and hence a new efficient adversarial attack. The method is first illustrated on a 2D toy example in order to visualize the neural network foliation and the corresponding attacks. Next, we report numerical results on the MNIST and CIFAR10 datasets with the proposed technique and state of the art attacks presented in Zhao et al. (2019) (OSSA) and Croce et al. (2020) (AutoAttack). The result show that the proposed attack is more efficient at all levels of available budget for the attack (norm of the attack), confirming that the curvature of the transverse neural network FIM foliation plays an important role in the robustness of neural networks. The main objective and interest of this study is to provide a mathematical understanding of the geometrical issues at play in the data space when constructing efficient attacks on neural networks.



## **19. Secure Control Systems for Autonomous Quadrotors against Cyber-Attacks**

cs.RO

The paper is based on an undergraduate thesis and is not intended for  publication in a journal

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.11897v1) [paper-pdf](http://arxiv.org/pdf/2409.11897v1)

**Authors**: Samuel Belkadi

**Abstract**: The problem of safety for robotic systems has been extensively studied. However, little attention has been given to security issues for three-dimensional systems, such as quadrotors. Malicious adversaries can compromise robot sensors and communication networks, causing incidents, achieving illegal objectives, or even injuring people. This study first designs an intelligent control system for autonomous quadrotors. Then, it investigates the problems of optimal false data injection attack scheduling and countermeasure design for unmanned aerial vehicles. Using a state-of-the-art deep learning-based approach, an optimal false data injection attack scheme is proposed to deteriorate a quadrotor's tracking performance with limited attack energy. Subsequently, an optimal tracking control strategy is learned to mitigate attacks and recover the quadrotor's tracking performance. We base our work on Agilicious, a state-of-the-art quadrotor recently deployed for autonomous settings. This paper is the first in the United Kingdom to deploy this quadrotor and implement reinforcement learning on its platform. Therefore, to promote easy reproducibility with minimal engineering overhead, we further provide (1) a comprehensive breakdown of this quadrotor, including software stacks and hardware alternatives; (2) a detailed reinforcement-learning framework to train autonomous controllers on Agilicious agents; and (3) a new open-source environment that builds upon PyFlyt for future reinforcement learning research on Agilicious platforms. Both simulated and real-world experiments are conducted to show the effectiveness of the proposed frameworks in section 5.2.



## **20. NPAT Null-Space Projected Adversarial Training Towards Zero Deterioration**

cs.LG

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.11754v1) [paper-pdf](http://arxiv.org/pdf/2409.11754v1)

**Authors**: Hanyi Hu, Qiao Han, Kui Chen, Yao Yang

**Abstract**: To mitigate the susceptibility of neural networks to adversarial attacks, adversarial training has emerged as a prevalent and effective defense strategy. Intrinsically, this countermeasure incurs a trade-off, as it sacrifices the model's accuracy in processing normal samples. To reconcile the trade-off, we pioneer the incorporation of null-space projection into adversarial training and propose two innovative Null-space Projection based Adversarial Training(NPAT) algorithms tackling sample generation and gradient optimization, named Null-space Projected Data Augmentation (NPDA) and Null-space Projected Gradient Descent (NPGD), to search for an overarching optimal solutions, which enhance robustness with almost zero deterioration in generalization performance. Adversarial samples and perturbations are constrained within the null-space of the decision boundary utilizing a closed-form null-space projector, effectively mitigating threat of attack stemming from unreliable features. Subsequently, we conducted experiments on the CIFAR10 and SVHN datasets and reveal that our methodology can seamlessly combine with adversarial training methods and obtain comparable robustness while keeping generalization close to a high-accuracy model.



## **21. Hard-Label Cryptanalytic Extraction of Neural Network Models**

cs.CR

Accepted by Asiacrypt 2024

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.11646v1) [paper-pdf](http://arxiv.org/pdf/2409.11646v1)

**Authors**: Yi Chen, Xiaoyang Dong, Jian Guo, Yantian Shen, Anyu Wang, Xiaoyun Wang

**Abstract**: The machine learning problem of extracting neural network parameters has been proposed for nearly three decades. Functionally equivalent extraction is a crucial goal for research on this problem. When the adversary has access to the raw output of neural networks, various attacks, including those presented at CRYPTO 2020 and EUROCRYPT 2024, have successfully achieved this goal. However, this goal is not achieved when neural networks operate under a hard-label setting where the raw output is inaccessible.   In this paper, we propose the first attack that theoretically achieves functionally equivalent extraction under the hard-label setting, which applies to ReLU neural networks. The effectiveness of our attack is validated through practical experiments on a wide range of ReLU neural networks, including neural networks trained on two real benchmarking datasets (MNIST, CIFAR10) widely used in computer vision. For a neural network consisting of $10^5$ parameters, our attack only requires several hours on a single core.



## **22. Image Hijacks: Adversarial Images can Control Generative Models at Runtime**

cs.LG

Project page at https://image-hijacks.github.io

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2309.00236v4) [paper-pdf](http://arxiv.org/pdf/2309.00236v4)

**Authors**: Luke Bailey, Euan Ong, Stuart Russell, Scott Emmons

**Abstract**: Are foundation models secure against malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control the behaviour of VLMs at inference time, and introduce the general Behaviour Matching algorithm for training image hijacks. From this, we derive the Prompt Matching method, allowing us to train hijacks matching the behaviour of an arbitrary user-defined text prompt (e.g. 'the Eiffel Tower is now located in Rome') using a generic, off-the-shelf dataset unrelated to our choice of prompt. We use Behaviour Matching to craft hijacks for four types of attack, forcing VLMs to generate outputs of the adversary's choice, leak information from their context window, override their safety training, and believe false statements. We study these attacks against LLaVA, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all attack types achieve a success rate of over 80%. Moreover, our attacks are automated and require only small image perturbations.



## **23. Golden Ratio Search: A Low-Power Adversarial Attack for Deep Learning based Modulation Classification**

cs.CR

5 pages, 1 figure, 3 tables

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.11454v1) [paper-pdf](http://arxiv.org/pdf/2409.11454v1)

**Authors**: Deepsayan Sadhukhan, Nitin Priyadarshini Shankar, Sheetal Kalyani

**Abstract**: We propose a minimal power white box adversarial attack for Deep Learning based Automatic Modulation Classification (AMC). The proposed attack uses the Golden Ratio Search (GRS) method to find powerful attacks with minimal power. We evaluate the efficacy of the proposed method by comparing it with existing adversarial attack approaches. Additionally, we test the robustness of the proposed attack against various state-of-the-art architectures, including defense mechanisms such as adversarial training, binarization, and ensemble methods. Experimental results demonstrate that the proposed attack is powerful, requires minimal power, and can be generated in less time, significantly challenging the resilience of current AMC methods.



## **24. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

cs.CR

24 pages

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.11295v1) [paper-pdf](http://arxiv.org/pdf/2409.11295v1)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have evolved rapidly and demonstrated remarkable potential. However, there are unprecedented safety risks associated with these them, which are nearly unexplored so far. In this work, we aim to narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a threat model that discusses the adversarial targets, constraints, and attack scenarios. Particularly, we consider two types of adversarial targets: stealing users' specific personally identifiable information (PII) or stealing the entire user request. To achieve these objectives, we propose a novel attack method, termed Environmental Injection Attack (EIA). This attack injects malicious content designed to adapt well to different environments where the agents operate, causing them to perform unintended actions. This work instantiates EIA specifically for the privacy scenario. It inserts malicious web elements alongside persuasive instructions that mislead web agents into leaking private information, and can further leverage CSS and JavaScript features to remain stealthy. We collect 177 actions steps that involve diverse PII categories on realistic websites from the Mind2Web dataset, and conduct extensive experiments using one of the most capable generalist web agent frameworks to date, SeeAct. The results demonstrate that EIA achieves up to 70% ASR in stealing users' specific PII. Stealing full user requests is more challenging, but a relaxed version of EIA can still achieve 16% ASR. Despite these concerning results, it is important to note that the attack can still be detectable through careful human inspection, highlighting a trade-off between high autonomy and security. This leads to our detailed discussion on the efficacy of EIA under different levels of human supervision as well as implications on defenses for generalist web agents.



## **25. Backdoor Attacks in Peer-to-Peer Federated Learning**

cs.LG

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2301.09732v4) [paper-pdf](http://arxiv.org/pdf/2301.09732v4)

**Authors**: Georgios Syros, Gokberk Yar, Simona Boboila, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Most machine learning applications rely on centralized learning processes, opening up the risk of exposure of their training datasets. While federated learning (FL) mitigates to some extent these privacy risks, it relies on a trusted aggregation server for training a shared global model. Recently, new distributed learning architectures based on Peer-to-Peer Federated Learning (P2PFL) offer advantages in terms of both privacy and reliability. Still, their resilience to poisoning attacks during training has not been investigated. In this paper, we propose new backdoor attacks for P2PFL that leverage structural graph properties to select the malicious nodes, and achieve high attack success, while remaining stealthy. We evaluate our attacks under various realistic conditions, including multiple graph topologies, limited adversarial visibility of the network, and clients with non-IID data. Finally, we show the limitations of existing defenses adapted from FL and design a new defense that successfully mitigates the backdoor attacks, without an impact on model accuracy.



## **26. A Survey of Machine Unlearning**

cs.LG

extend the survey with more recent published work and add more  discussions

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2209.02299v6) [paper-pdf](http://arxiv.org/pdf/2209.02299v6)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Zhao Ren, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstract**: Today, computer systems hold large amounts of personal data. Yet while such an abundance of data allows breakthroughs in artificial intelligence, and especially machine learning (ML), its existence can be a threat to user privacy, and it can weaken the bonds of trust between humans and AI. Recent regulations now require that, on request, private information about a user must be removed from both computer systems and from ML models, i.e. ``the right to be forgotten''). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often `remember' the old data. Contemporary adversarial attacks on trained models have proven that we can learn whether an instance or an attribute belonged to the training data. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to completely solve the problem due to the lack of common frameworks and resources. Therefore, this paper aspires to present a comprehensive examination of machine unlearning's concepts, scenarios, methods, and applications. Specifically, as a category collection of cutting-edge studies, the intention behind this article is to serve as a comprehensive resource for researchers and practitioners seeking an introduction to machine unlearning and its formulations, design criteria, removal requests, algorithms, and applications. In addition, we aim to highlight the key findings, current trends, and new research areas that have not yet featured the use of machine unlearning but could benefit greatly from it. We hope this survey serves as a valuable resource for ML researchers and those seeking to innovate privacy technologies. Our resources are publicly available at https://github.com/tamlhp/awesome-machine-unlearning.



## **27. Remote Keylogging Attacks in Multi-user VR Applications**

cs.CR

Accepted for Usenix 2024

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2405.14036v2) [paper-pdf](http://arxiv.org/pdf/2405.14036v2)

**Authors**: Zihao Su, Kunlin Cai, Reuben Beeler, Lukas Dresel, Allan Garcia, Ilya Grishchenko, Yuan Tian, Christopher Kruegel, Giovanni Vigna

**Abstract**: As Virtual Reality (VR) applications grow in popularity, they have bridged distances and brought users closer together. However, with this growth, there have been increasing concerns about security and privacy, especially related to the motion data used to create immersive experiences. In this study, we highlight a significant security threat in multi-user VR applications, which are applications that allow multiple users to interact with each other in the same virtual space. Specifically, we propose a remote attack that utilizes the avatar rendering information collected from an adversary's game clients to extract user-typed secrets like credit card information, passwords, or private conversations. We do this by (1) extracting motion data from network packets, and (2) mapping motion data to keystroke entries. We conducted a user study to verify the attack's effectiveness, in which our attack successfully inferred 97.62% of the keystrokes. Besides, we performed an additional experiment to underline that our attack is practical, confirming its effectiveness even when (1) there are multiple users in a room, and (2) the attacker cannot see the victims. Moreover, we replicated our proposed attack on four applications to demonstrate the generalizability of the attack. Lastly, we proposed a defense against the attack, which has been implemented by major players in the VR industry. These results underscore the severity of the vulnerability and its potential impact on millions of VR social platform users.



## **28. An Anti-disguise Authentication System Using the First Impression of Avatar in Metaverse**

cs.CR

19 pages, 16 figures

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.10850v1) [paper-pdf](http://arxiv.org/pdf/2409.10850v1)

**Authors**: Zhenyong Zhang, Kedi Yang, Youliang Tian, Jianfeng Ma

**Abstract**: Metaverse is a vast virtual world parallel to the physical world, where the user acts as an avatar to enjoy various services that break through the temporal and spatial limitations of the physical world. Metaverse allows users to create arbitrary digital appearances as their own avatars by which an adversary may disguise his/her avatar to fraud others. In this paper, we propose an anti-disguise authentication method that draws on the idea of the first impression from the physical world to recognize an old friend. Specifically, the first meeting scenario in the metaverse is stored and recalled to help the authentication between avatars. To prevent the adversary from replacing and forging the first impression, we construct a chameleon-based signcryption mechanism and design a ciphertext authentication protocol to ensure the public verifiability of encrypted identities. The security analysis shows that the proposed signcryption mechanism meets not only the security requirement but also the public verifiability. Besides, the ciphertext authentication protocol has the capability of defending against the replacing and forging attacks on the first impression. Extensive experiments show that the proposed avatar authentication system is able to achieve anti-disguise authentication at a low storage consumption on the blockchain.



## **29. Weak Superimposed Codes of Improved Asymptotic Rate and Their Randomized Construction**

cs.IT

6 pages, accepted for presentation at the 2022 IEEE International  Symposium on Information Theory (ISIT)

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2409.10511v1) [paper-pdf](http://arxiv.org/pdf/2409.10511v1)

**Authors**: Yu Tsunoda, Yuichiro Fujiwara

**Abstract**: Weak superimposed codes are combinatorial structures related closely to generalized cover-free families, superimposed codes, and disjunct matrices in that they are only required to satisfy similar but less stringent conditions. This class of codes may also be seen as a stricter variant of what are known as locally thin families in combinatorics. Originally, weak superimposed codes were introduced in the context of multimedia content protection against illegal distribution of copies under the assumption that a coalition of malicious users may employ the averaging attack with adversarial noise. As in many other kinds of codes in information theory, it is of interest and importance in the study of weak superimposed codes to find the highest achievable rate in the asymptotic regime and give an efficient construction that produces an infinite sequence of codes that achieve it. Here, we prove a tighter lower bound than the sharpest known one on the rate of optimal weak superimposed codes and give a polynomial-time randomized construction algorithm for codes that asymptotically attain our improved bound with high probability. Our probabilistic approach is versatile and applicable to many other related codes and arrays.



## **30. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

cs.CL

28 pages incl. appendix, updated version

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2402.10527v2) [paper-pdf](http://arxiv.org/pdf/2402.10527v2)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.



## **31. Deep Reinforcement Learning for Autonomous Cyber Operations: A Survey**

cs.LG

89 pages, 14 figures, 4 tables

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2310.07745v2) [paper-pdf](http://arxiv.org/pdf/2310.07745v2)

**Authors**: Gregory Palmer, Chris Parry, Daniel J. B. Harrold, Chris Willis

**Abstract**: The rapid increase in the number of cyber-attacks in recent years raises the need for principled methods for defending networks against malicious actors. Deep reinforcement learning (DRL) has emerged as a promising approach for mitigating these attacks. However, while DRL has shown much potential for cyber defence, numerous challenges must be overcome before DRL can be applied to autonomous cyber operations (ACO) at scale. Principled methods are required for environments that confront learners with very high-dimensional state spaces, large multi-discrete action spaces, and adversarial learning. Recent works have reported success in solving these problems individually. There have also been impressive engineering efforts towards solving all three for real-time strategy games. However, applying DRL to the full ACO problem remains an open challenge. Here, we survey the relevant DRL literature and conceptualize an idealised ACO-DRL agent. We provide: i.) A summary of the domain properties that define the ACO problem; ii.) A comprehensive comparison of current ACO environments used for benchmarking DRL approaches; iii.) An overview of state-of-the-art approaches for scaling DRL to domains that confront learners with the curse of dimensionality, and; iv.) A survey and critique of current methods for limiting the exploitability of agents within adversarial settings from the perspective of ACO. We conclude with open research questions that we hope will motivate future directions for researchers and practitioners working on ACO.



## **32. Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents**

cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.05870v2) [paper-pdf](http://arxiv.org/pdf/2406.05870v2)

**Authors**: Avital Shafran, Roei Schuster, Vitaly Shmatikov

**Abstract**: Retrieval-augmented generation (RAG) systems respond to queries by retrieving relevant documents from a knowledge database, then generating an answer by applying an LLM to the retrieved documents. We demonstrate that RAG systems that operate on databases with untrusted content are vulnerable to a new class of denial-of-service attacks we call jamming. An adversary can add a single ``blocker'' document to the database that will be retrieved in response to a specific query and result in the RAG system not answering this query - ostensibly because it lacks the information or because the answer is unsafe.   We describe and measure the efficacy of several methods for generating blocker documents, including a new method based on black-box optimization. This method (1) does not rely on instruction injection, (2) does not require the adversary to know the embedding or LLM used by the target RAG system, and (3) does not use an auxiliary LLM to generate blocker documents.   We evaluate jamming attacks on several LLMs and embeddings and demonstrate that the existing safety metrics for LLMs do not capture their vulnerability to jamming. We then discuss defenses against blocker documents.



## **33. Towards Evaluating the Robustness of Visual State Space Models**

cs.CV

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.09407v2) [paper-pdf](http://arxiv.org/pdf/2406.09407v2)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Vision State Space Models (VSSMs), a novel architecture that combines the strengths of recurrent neural networks and latent variable models, have demonstrated remarkable performance in visual perception tasks by efficiently capturing long-range dependencies and modeling complex visual dynamics. However, their robustness under natural and adversarial perturbations remains a critical concern. In this work, we present a comprehensive evaluation of VSSMs' robustness under various perturbation scenarios, including occlusions, image structure, common corruptions, and adversarial attacks, and compare their performance to well-established architectures such as transformers and Convolutional Neural Networks. Furthermore, we investigate the resilience of VSSMs to object-background compositional changes on sophisticated benchmarks designed to test model performance in complex visual scenes. We also assess their robustness on object detection and segmentation tasks using corrupted datasets that mimic real-world scenarios. To gain a deeper understanding of VSSMs' adversarial robustness, we conduct a frequency-based analysis of adversarial attacks, evaluating their performance against low-frequency and high-frequency perturbations. Our findings highlight the strengths and limitations of VSSMs in handling complex visual corruptions, offering valuable insights for future research. Our code and models will be available at https://github.com/HashmatShadab/MambaRobustness.



## **34. Multi-agent Attacks for Black-box Social Recommendations**

cs.SI

Accepted by ACM TOIS

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2311.07127v4) [paper-pdf](http://arxiv.org/pdf/2311.07127v4)

**Authors**: Shijie Wang, Wenqi Fan, Xiao-yong Wei, Xiaowei Mei, Shanru Lin, Qing Li

**Abstract**: The rise of online social networks has facilitated the evolution of social recommender systems, which incorporate social relations to enhance users' decision-making process. With the great success of Graph Neural Networks (GNNs) in learning node representations, GNN-based social recommendations have been widely studied to model user-item interactions and user-user social relations simultaneously. Despite their great successes, recent studies have shown that these advanced recommender systems are highly vulnerable to adversarial attacks, in which attackers can inject well-designed fake user profiles to disrupt recommendation performances. While most existing studies mainly focus on argeted attacks to promote target items on vanilla recommender systems, untargeted attacks to degrade the overall prediction performance are less explored on social recommendations under a black-box scenario. To perform untargeted attacks on social recommender systems, attackers can construct malicious social relationships for fake users to enhance the attack performance. However, the coordination of social relations and item profiles is challenging for attacking black-box social recommendations. To address this limitation, we first conduct several preliminary studies to demonstrate the effectiveness of cross-community connections and cold-start items in degrading recommendations performance. Specifically, we propose a novel framework MultiAttack based on multi-agent reinforcement learning to coordinate the generation of cold-start item profiles and cross-community social relations for conducting untargeted attacks on black-box social recommendations. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of our proposed attacking framework under the black-box setting.



## **35. Towards Adversarial Robustness And Backdoor Mitigation in SSL**

cs.CV

8 pages, 2 figures

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2403.15918v3) [paper-pdf](http://arxiv.org/pdf/2403.15918v3)

**Authors**: Aryan Satpathy, Nilaksh Singh, Dhruva Rajwade, Somesh Kumar

**Abstract**: Self-Supervised Learning (SSL) has shown great promise in learning representations from unlabeled data. The power of learning representations without the need for human annotations has made SSL a widely used technique in real-world problems. However, SSL methods have recently been shown to be vulnerable to backdoor attacks, where the learned model can be exploited by adversaries to manipulate the learned representations, either through tampering the training data distribution, or via modifying the model itself. This work aims to address defending against backdoor attacks in SSL, where the adversary has access to a realistic fraction of the SSL training data, and no access to the model. We use novel methods that are computationally efficient as well as generalizable across different problem settings. We also investigate the adversarial robustness of SSL models when trained with our method, and show insights into increased robustness in SSL via frequency domain augmentations. We demonstrate the effectiveness of our method on a variety of SSL benchmarks, and show that our method is able to mitigate backdoor attacks while maintaining high performance on downstream tasks. Code for our work is available at github.com/Aryan-Satpathy/Backdoor



## **36. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

cs.CR

(Last update!, a constructive comment from arxiv led to this latest  update ) Stochastic investment models and a Bayesian approach to better  modeling of uncertainty : adversarial machine learning or Stochastic market.  arXiv admin note: substantial text overlap with arXiv:2402.05967 (see this  link to the paper by : Orson Mengara)

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.10719v4) [paper-pdf](http://arxiv.org/pdf/2406.10719v4)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.



## **37. Exact Recovery Guarantees for Parameterized Non-linear System Identification Problem under Adversarial Attacks**

math.OC

33 pages

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2409.00276v2) [paper-pdf](http://arxiv.org/pdf/2409.00276v2)

**Authors**: Haixiang Zhang, Baturalp Yalcin, Javad Lavaei, Eduardo D. Sontag

**Abstract**: In this work, we study the system identification problem for parameterized non-linear systems using basis functions under adversarial attacks. Motivated by the LASSO-type estimators, we analyze the exact recovery property of a non-smooth estimator, which is generated by solving an embedded $\ell_1$-loss minimization problem. First, we derive necessary and sufficient conditions for the well-specifiedness of the estimator and the uniqueness of global solutions to the underlying optimization problem. Next, we provide exact recovery guarantees for the estimator under two different scenarios of boundedness and Lipschitz continuity of the basis functions. The non-asymptotic exact recovery is guaranteed with high probability, even when there are more severely corrupted data than clean data. Finally, we numerically illustrate the validity of our theory. This is the first study on the sample complexity analysis of a non-smooth estimator for the non-linear system identification problem.



## **38. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.04755v2) [paper-pdf](http://arxiv.org/pdf/2406.04755v2)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.



## **39. Revisiting Physical-World Adversarial Attack on Traffic Sign Recognition: A Commercial Systems Perspective**

cs.CR

Accepted by NDSS 2025

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09860v1) [paper-pdf](http://arxiv.org/pdf/2409.09860v1)

**Authors**: Ningfei Wang, Shaoyuan Xie, Takami Sato, Yunpeng Luo, Kaidi Xu, Qi Alfred Chen

**Abstract**: Traffic Sign Recognition (TSR) is crucial for safe and correct driving automation. Recent works revealed a general vulnerability of TSR models to physical-world adversarial attacks, which can be low-cost, highly deployable, and capable of causing severe attack effects such as hiding a critical traffic sign or spoofing a fake one. However, so far existing works generally only considered evaluating the attack effects on academic TSR models, leaving the impacts of such attacks on real-world commercial TSR systems largely unclear. In this paper, we conduct the first large-scale measurement of physical-world adversarial attacks against commercial TSR systems. Our testing results reveal that it is possible for existing attack works from academia to have highly reliable (100\%) attack success against certain commercial TSR system functionality, but such attack capabilities are not generalizable, leading to much lower-than-expected attack success rates overall. We find that one potential major factor is a spatial memorization design that commonly exists in today's commercial TSR systems. We design new attack success metrics that can mathematically model the impacts of such design on the TSR system-level attack success, and use them to revisit existing attacks. Through these efforts, we uncover 7 novel observations, some of which directly challenge the observations or claims in prior works due to the introduction of the new metrics.



## **40. Federated Learning in Adversarial Environments: Testbed Design and Poisoning Resilience in Cybersecurity**

cs.CR

7 pages, 4 figures

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09794v1) [paper-pdf](http://arxiv.org/pdf/2409.09794v1)

**Authors**: Hao Jian Huang, Bekzod Iskandarov, Mizanur Rahman, Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: This paper presents the design and implementation of a Federated Learning (FL) testbed, focusing on its application in cybersecurity and evaluating its resilience against poisoning attacks. Federated Learning allows multiple clients to collaboratively train a global model while keeping their data decentralized, addressing critical needs for data privacy and security, particularly in sensitive fields like cybersecurity. Our testbed, built using the Flower framework, facilitates experimentation with various FL frameworks, assessing their performance, scalability, and ease of integration. Through a case study on federated intrusion detection systems, we demonstrate the testbed's capabilities in detecting anomalies and securing critical infrastructure without exposing sensitive network data. Comprehensive poisoning tests, targeting both model and data integrity, evaluate the system's robustness under adversarial conditions. Our results show that while federated learning enhances data privacy and distributed learning, it remains vulnerable to poisoning attacks, which must be mitigated to ensure its reliability in real-world applications.



## **41. Real-world Adversarial Defense against Patch Attacks based on Diffusion Model**

cs.CV

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2409.09406v1) [paper-pdf](http://arxiv.org/pdf/2409.09406v1)

**Authors**: Xingxing Wei, Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su

**Abstract**: Adversarial patches present significant challenges to the robustness of deep learning models, making the development of effective defenses become critical for real-world applications. This paper introduces DIFFender, a novel DIFfusion-based DeFender framework that leverages the power of a text-guided diffusion model to counter adversarial patch attacks. At the core of our approach is the discovery of the Adversarial Anomaly Perception (AAP) phenomenon, which enables the diffusion model to accurately detect and locate adversarial patches by analyzing distributional anomalies. DIFFender seamlessly integrates the tasks of patch localization and restoration within a unified diffusion model framework, enhancing defense efficacy through their close interaction. Additionally, DIFFender employs an efficient few-shot prompt-tuning algorithm, facilitating the adaptation of the pre-trained diffusion model to defense tasks without the need for extensive retraining. Our comprehensive evaluation, covering image classification and face recognition tasks, as well as real-world scenarios, demonstrates DIFFender's robust performance against adversarial attacks. The framework's versatility and generalizability across various settings, classifiers, and attack methodologies mark a significant advancement in adversarial patch defense strategies. Except for the popular visible domain, we have identified another advantage of DIFFender: its capability to easily expand into the infrared domain. Consequently, we demonstrate the good flexibility of DIFFender, which can defend against both infrared and visible adversarial patch attacks alternatively using a universal defense framework.



## **42. Regret-Optimal Defense Against Stealthy Adversaries: A System Level Approach**

eess.SY

Accepted, IEEE Conference on Decision and Control (CDC), 2024

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2407.18448v2) [paper-pdf](http://arxiv.org/pdf/2407.18448v2)

**Authors**: Hiroyasu Tsukamoto, Joudi Hajar, Soon-Jo Chung, Fred Y. Hadaegh

**Abstract**: Modern control designs in robotics, aerospace, and cyber-physical systems rely heavily on real-world data obtained through system outputs. However, these outputs can be compromised by system faults and malicious attacks, distorting critical system information needed for secure and reliable operation. In this paper, we introduce a novel regret-optimal control framework for designing controllers that make a linear system robust against stealthy attacks, including both sensor and actuator attacks. Specifically, we present (a) a convex optimization-based system metric to quantify the regret under the worst-case stealthy attack (the difference between actual performance and optimal performance with hindsight of the attack), which adapts and improves upon the $\mathcal{H}_2$ and $\mathcal{H}_{\infty}$ norms in the presence of stealthy adversaries, (b) an optimization problem for minimizing the regret of (a) in system-level parameterization, enabling localized and distributed implementation in large-scale systems, and (c) a rank-constrained optimization problem equivalent to the optimization of (b), which can be solved using convex rank minimization methods. We also present numerical simulations that demonstrate the effectiveness of our proposed framework.



## **43. Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness**

cs.CL

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.04585v3) [paper-pdf](http://arxiv.org/pdf/2408.04585v3)

**Authors**: Xiaojing Fan, Chunliang Tao

**Abstract**: With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs and conduct extensive experiments on three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.



## **44. Tamper-Resistant Safeguards for Open-Weight LLMs**

cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.00761v3) [paper-pdf](http://arxiv.org/pdf/2408.00761v3)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.



## **45. Eliminating Catastrophic Overfitting Via Abnormal Adversarial Examples Regularization**

cs.LG

Accepted by NeurIPS 2023

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2404.08154v2) [paper-pdf](http://arxiv.org/pdf/2404.08154v2)

**Authors**: Runqi Lin, Chaojian Yu, Tongliang Liu

**Abstract**: Single-step adversarial training (SSAT) has demonstrated the potential to achieve both efficiency and robustness. However, SSAT suffers from catastrophic overfitting (CO), a phenomenon that leads to a severely distorted classifier, making it vulnerable to multi-step adversarial attacks. In this work, we observe that some adversarial examples generated on the SSAT-trained network exhibit anomalous behaviour, that is, although these training samples are generated by the inner maximization process, their associated loss decreases instead, which we named abnormal adversarial examples (AAEs). Upon further analysis, we discover a close relationship between AAEs and classifier distortion, as both the number and outputs of AAEs undergo a significant variation with the onset of CO. Given this observation, we re-examine the SSAT process and uncover that before the occurrence of CO, the classifier already displayed a slight distortion, indicated by the presence of few AAEs. Furthermore, the classifier directly optimizing these AAEs will accelerate its distortion, and correspondingly, the variation of AAEs will sharply increase as a result. In such a vicious circle, the classifier rapidly becomes highly distorted and manifests as CO within a few iterations. These observations motivate us to eliminate CO by hindering the generation of AAEs. Specifically, we design a novel method, termed Abnormal Adversarial Examples Regularization (AAER), which explicitly regularizes the variation of AAEs to hinder the classifier from becoming distorted. Extensive experiments demonstrate that our method can effectively eliminate CO and further boost adversarial robustness with negligible additional computational overhead.



## **46. Layer-Aware Analysis of Catastrophic Overfitting: Revealing the Pseudo-Robust Shortcut Dependency**

cs.LG

Accepted by ICML 2024

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2405.16262v2) [paper-pdf](http://arxiv.org/pdf/2405.16262v2)

**Authors**: Runqi Lin, Chaojian Yu, Bo Han, Hang Su, Tongliang Liu

**Abstract**: Catastrophic overfitting (CO) presents a significant challenge in single-step adversarial training (AT), manifesting as highly distorted deep neural networks (DNNs) that are vulnerable to multi-step adversarial attacks. However, the underlying factors that lead to the distortion of decision boundaries remain unclear. In this work, we delve into the specific changes within different DNN layers and discover that during CO, the former layers are more susceptible, experiencing earlier and greater distortion, while the latter layers show relative insensitivity. Our analysis further reveals that this increased sensitivity in former layers stems from the formation of pseudo-robust shortcuts, which alone can impeccably defend against single-step adversarial attacks but bypass genuine-robust learning, resulting in distorted decision boundaries. Eliminating these shortcuts can partially restore robustness in DNNs from the CO state, thereby verifying that dependence on them triggers the occurrence of CO. This understanding motivates us to implement adaptive weight perturbations across different layers to hinder the generation of pseudo-robust shortcuts, consequently mitigating CO. Extensive experiments demonstrate that our proposed method, Layer-Aware Adversarial Weight Perturbation (LAP), can effectively prevent CO and further enhance robustness.



## **47. Cybersecurity Software Tool Evaluation Using a 'Perfect' Network Model**

cs.CR

The U.S. federal sponsor has requested that we not include funding  acknowledgement for this publication

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.09175v1) [paper-pdf](http://arxiv.org/pdf/2409.09175v1)

**Authors**: Jeremy Straub

**Abstract**: Cybersecurity software tool evaluation is difficult due to the inherently adversarial nature of the field. A penetration testing (or offensive) tool must be tested against a viable defensive adversary and a defensive tool must, similarly, be tested against a viable offensive adversary. Characterizing the tool's performance inherently depends on the quality of the adversary, which can vary from test to test. This paper proposes the use of a 'perfect' network, representing computing systems, a network and the attack pathways through it as a methodology to use for testing cybersecurity decision-making tools. This facilitates testing by providing a known and consistent standard for comparison. It also allows testing to include researcher-selected levels of error, noise and uncertainty to evaluate cybersecurity tools under these experimental conditions.



## **48. Clean Label Attacks against SLU Systems**

cs.CR

Accepted at IEEE SLT 2024

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.08985v1) [paper-pdf](http://arxiv.org/pdf/2409.08985v1)

**Authors**: Henry Li Xinyuan, Sonal Joshi, Thomas Thebaud, Jesus Villalba, Najim Dehak, Sanjeev Khudanpur

**Abstract**: Poisoning backdoor attacks involve an adversary manipulating the training data to induce certain behaviors in the victim model by inserting a trigger in the signal at inference time. We adapted clean label backdoor (CLBD)-data poisoning attacks, which do not modify the training labels, on state-of-the-art speech recognition models that support/perform a Spoken Language Understanding task, achieving 99.8% attack success rate by poisoning 10% of the training data. We analyzed how varying the signal-strength of the poison, percent of samples poisoned, and choice of trigger impact the attack. We also found that CLBD attacks are most successful when applied to training samples that are inherently hard for a proxy model. Using this strategy, we achieved an attack success rate of 99.3% by poisoning a meager 1.5% of the training data. Finally, we applied two previously developed defenses against gradient-based attacks, and found that they attain mixed success against poisoning.



## **49. XSub: Explanation-Driven Adversarial Attack against Blackbox Classifiers via Feature Substitution**

cs.LG

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.08919v1) [paper-pdf](http://arxiv.org/pdf/2409.08919v1)

**Authors**: Kiana Vu, Phung Lai, Truc Nguyen

**Abstract**: Despite its significant benefits in enhancing the transparency and trustworthiness of artificial intelligence (AI) systems, explainable AI (XAI) has yet to reach its full potential in real-world applications. One key challenge is that XAI can unintentionally provide adversaries with insights into black-box models, inevitably increasing their vulnerability to various attacks. In this paper, we develop a novel explanation-driven adversarial attack against black-box classifiers based on feature substitution, called XSub. The key idea of XSub is to strategically replace important features (identified via XAI) in the original sample with corresponding important features from a "golden sample" of a different label, thereby increasing the likelihood of the model misclassifying the perturbed sample. The degree of feature substitution is adjustable, allowing us to control how much of the original samples information is replaced. This flexibility effectively balances a trade-off between the attacks effectiveness and its stealthiness. XSub is also highly cost-effective in that the number of required queries to the prediction model and the explanation model in conducting the attack is in O(1). In addition, XSub can be easily extended to launch backdoor attacks in case the attacker has access to the models training data. Our evaluation demonstrates that XSub is not only effective and stealthy but also cost-effective, enabling its application across a wide range of AI models.



## **50. Are Existing Road Design Guidelines Suitable for Autonomous Vehicles?**

cs.CV

Currently under review by IEEE Transactions on Software Engineering  (TSE)

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.10562v1) [paper-pdf](http://arxiv.org/pdf/2409.10562v1)

**Authors**: Yang Sun, Christopher M. Poskitt, Jun Sun

**Abstract**: The emergence of Autonomous Vehicles (AVs) has spurred research into testing the resilience of their perception systems, i.e. to ensure they are not susceptible to making critical misjudgements. It is important that they are tested not only with respect to other vehicles on the road, but also those objects placed on the roadside. Trash bins, billboards, and greenery are all examples of such objects, typically placed according to guidelines that were developed for the human visual system, and which may not align perfectly with the needs of AVs. Existing tests, however, usually focus on adversarial objects with conspicuous shapes/patches, that are ultimately unrealistic given their unnatural appearances and the need for white box knowledge. In this work, we introduce a black box attack on the perception systems of AVs, in which the objective is to create realistic adversarial scenarios (i.e. satisfying road design guidelines) by manipulating the positions of common roadside objects, and without resorting to `unnatural' adversarial patches. In particular, we propose TrashFuzz , a fuzzing algorithm to find scenarios in which the placement of these objects leads to substantial misperceptions by the AV -- such as mistaking a traffic light's colour -- with overall the goal of causing it to violate traffic laws. To ensure the realism of these scenarios, they must satisfy several rules encoding regulatory guidelines about the placement of objects on public streets. We implemented and evaluated these attacks for the Apollo, finding that TrashFuzz induced it into violating 15 out of 24 different traffic laws.



