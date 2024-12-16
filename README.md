# Latest Adversarial Attack Papers
**update at 2024-12-16 09:59:59**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. A Semi Black-Box Adversarial Bit-Flip Attack with Limited DNN Model Information**

cs.CR

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2412.09450v1) [paper-pdf](http://arxiv.org/pdf/2412.09450v1)

**Authors**: Behnam Ghavami, Mani Sadati, Mohammad Shahidzadeh, Lesley Shannon, Steve Wilton

**Abstract**: Despite the rising prevalence of deep neural networks (DNNs) in cyber-physical systems, their vulnerability to adversarial bit-flip attacks (BFAs) is a noteworthy concern. This paper proposes B3FA, a semi-black-box BFA-based parameter attack on DNNs, assuming the adversary has limited knowledge about the model. We consider practical scenarios often feature a more restricted threat model for real-world systems, contrasting with the typical BFA models that presuppose the adversary's full access to a network's inputs and parameters. The introduced bit-flip approach utilizes a magnitude-based ranking method and a statistical re-construction technique to identify the vulnerable bits. We demonstrate the effectiveness of B3FA on several DNN models in a semi-black-box setting. For example, B3FA could drop the accuracy of a MobileNetV2 from 69.84% to 9% with only 20 bit-flips in a real-world setting.



## **2. On the Robustness of Kolmogorov-Arnold Networks: An Adversarial Perspective**

cs.CV

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2408.13809v2) [paper-pdf](http://arxiv.org/pdf/2408.13809v2)

**Authors**: Tal Alter, Raz Lapid, Moshe Sipper

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently emerged as a novel approach to function approximation, demonstrating remarkable potential in various domains. Despite their theoretical promise, the robustness of KANs under adversarial conditions has yet to be thoroughly examined. In this paper we explore the adversarial robustness of KANs, with a particular focus on image classification tasks. We assess the performance of KANs against standard white box and black-box adversarial attacks, comparing their resilience to that of established neural network architectures. Our experimental evaluation encompasses a variety of standard image classification benchmark datasets and investigates both fully connected and convolutional neural network architectures, of three sizes: small, medium, and large. We conclude that small- and medium-sized KANs (either fully connected or convolutional) are not consistently more robust than their standard counterparts, but that large-sized KANs are, by and large, more robust. This comprehensive evaluation of KANs in adversarial scenarios offers the first in-depth analysis of KAN security, laying the groundwork for future research in this emerging field.



## **3. FedAA: A Reinforcement Learning Perspective on Adaptive Aggregation for Fair and Robust Federated Learning**

cs.LG

AAAI 2025

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2402.05541v2) [paper-pdf](http://arxiv.org/pdf/2402.05541v2)

**Authors**: Jialuo He, Wei Chen, Xiaojin Zhang

**Abstract**: Federated Learning (FL) has emerged as a promising approach for privacy-preserving model training across decentralized devices. However, it faces challenges such as statistical heterogeneity and susceptibility to adversarial attacks, which can impact model robustness and fairness. Personalized FL attempts to provide some relief by customizing models for individual clients. However, it falls short in addressing server-side aggregation vulnerabilities. We introduce a novel method called \textbf{FedAA}, which optimizes client contributions via \textbf{A}daptive \textbf{A}ggregation to enhance model robustness against malicious clients and ensure fairness across participants in non-identically distributed settings. To achieve this goal, we propose an approach involving a Deep Deterministic Policy Gradient-based algorithm for continuous control of aggregation weights, an innovative client selection method based on model parameter distances, and a reward mechanism guided by validation set performance. Empirically, extensive experiments demonstrate that, in terms of robustness, \textbf{FedAA} outperforms the state-of-the-art methods, while maintaining comparable levels of fairness, offering a promising solution to build resilient and fair federated systems. Our code is available at https://github.com/Gp1g/FedAA.



## **4. On the Generation and Removal of Speaker Adversarial Perturbation for Voice-Privacy Protection**

cs.SD

6 pages, 3 figures, published to IEEE SLT Workshop 2024

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2412.09195v1) [paper-pdf](http://arxiv.org/pdf/2412.09195v1)

**Authors**: Chenyang Guo, Liping Chen, Zhuhai Li, Kong Aik Lee, Zhen-Hua Ling, Wu Guo

**Abstract**: Neural networks are commonly known to be vulnerable to adversarial attacks mounted through subtle perturbation on the input data. Recent development in voice-privacy protection has shown the positive use cases of the same technique to conceal speaker's voice attribute with additive perturbation signal generated by an adversarial network. This paper examines the reversibility property where an entity generating the adversarial perturbations is authorized to remove them and restore original speech (e.g., the speaker him/herself). A similar technique could also be used by an investigator to deanonymize a voice-protected speech to restore criminals' identities in security and forensic analysis. In this setting, the perturbation generative module is assumed to be known in the removal process. To this end, a joint training of perturbation generation and removal modules is proposed. Experimental results on the LibriSpeech dataset demonstrated that the subtle perturbations added to the original speech can be predicted from the anonymized speech while achieving the goal of privacy protection. By removing these perturbations from the anonymized sample, the original speech can be restored. Audio samples can be found in \url{https://voiceprivacy.github.io/Perturbation-Generation-Removal/}.



## **5. Evaluating Adversarial Attacks on Traffic Sign Classifiers beyond Standard Baselines**

cs.CV

Accepted for publication at ICMLA 2024

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2412.09150v1) [paper-pdf](http://arxiv.org/pdf/2412.09150v1)

**Authors**: Svetlana Pavlitska, Leopold Müller, J. Marius Zöllner

**Abstract**: Adversarial attacks on traffic sign classification models were among the first successfully tried in the real world. Since then, the research in this area has been mainly restricted to repeating baseline models, such as LISA-CNN or GTSRB-CNN, and similar experiment settings, including white and black patches on traffic signs. In this work, we decouple model architectures from the datasets and evaluate on further generic models to make a fair comparison. Furthermore, we compare two attack settings, inconspicuous and visible, which are usually regarded without direct comparison. Our results show that standard baselines like LISA-CNN or GTSRB-CNN are significantly more susceptible than the generic ones. We, therefore, suggest evaluating new attacks on a broader spectrum of baselines in the future. Our code is available at \url{https://github.com/KASTEL-MobilityLab/attacks-on-traffic-sign-recognition/}.



## **6. Unlearning or Concealment? A Critical Analysis and Evaluation Metrics for Unlearning in Diffusion Models**

cs.LG

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2409.05668v2) [paper-pdf](http://arxiv.org/pdf/2409.05668v2)

**Authors**: Aakash Sen Sharma, Niladri Sarkar, Vikram Chundawat, Ankur A Mali, Murari Mandal

**Abstract**: Recent research has seen significant interest in methods for concept removal and targeted forgetting in text-to-image diffusion models. In this paper, we conduct a comprehensive white-box analysis showing the vulnerabilities in existing diffusion model unlearning methods. We show that existing unlearning methods lead to decoupling of the targeted concepts (meant to be forgotten) for the corresponding prompts. This is concealment and not actual forgetting, which was the original goal. This paper presents a rigorous theoretical and empirical examination of five commonly used techniques for unlearning in diffusion models, while showing their potential weaknesses. We introduce two new evaluation metrics: Concept Retrieval Score (\textbf{CRS}) and Concept Confidence Score (\textbf{CCS}). These metrics are based on a successful adversarial attack setup that can recover \textit{forgotten} concepts from unlearned diffusion models. \textbf{CRS} measures the similarity between the latent representations of the unlearned and fully trained models after unlearning. It reports the extent of retrieval of the \textit{forgotten} concepts with increasing amount of guidance. CCS quantifies the confidence of the model in assigning the target concept to the manipulated data. It reports the probability of the \textit{unlearned} model's generations to be aligned with the original domain knowledge with increasing amount of guidance. The \textbf{CCS} and \textbf{CRS} enable a more robust evaluation of concept erasure methods. Evaluating existing five state-of-the-art methods with our metrics, reveal significant shortcomings in their ability to truly \textit{unlearn}. Source Code: \color{blue}{https://respailab.github.io/unlearning-or-concealment}



## **7. Deep Learning Model Security: Threats and Defenses**

cs.CR

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2412.08969v1) [paper-pdf](http://arxiv.org/pdf/2412.08969v1)

**Authors**: Tianyang Wang, Ziqian Bi, Yichao Zhang, Ming Liu, Weiche Hsieh, Pohsun Feng, Lawrence K. Q. Yan, Yizhu Wen, Benji Peng, Junyu Liu, Keyu Chen, Sen Zhang, Ming Li, Chuanqi Jiang, Xinyuan Song, Junjie Yang, Bowen Jing, Jintao Ren, Junhao Song, Hong-Ming Tseng, Silin Chen, Yunze Wang, Chia Xin Liang, Jiawei Xu, Xuanhe Pan, Jinlang Wang, Qian Niu

**Abstract**: Deep learning has transformed AI applications but faces critical security challenges, including adversarial attacks, data poisoning, model theft, and privacy leakage. This survey examines these vulnerabilities, detailing their mechanisms and impact on model integrity and confidentiality. Practical implementations, including adversarial examples, label flipping, and backdoor attacks, are explored alongside defenses such as adversarial training, differential privacy, and federated learning, highlighting their strengths and limitations.   Advanced methods like contrastive and self-supervised learning are presented for enhancing robustness. The survey concludes with future directions, emphasizing automated defenses, zero-trust architectures, and the security challenges of large AI models. A balanced approach to performance and security is essential for developing reliable deep learning systems.



## **8. Respect the model: Fine-grained and Robust Explanation with Sharing Ratio Decomposition**

cs.CV

To be published in ICLR 2024

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2402.03348v2) [paper-pdf](http://arxiv.org/pdf/2402.03348v2)

**Authors**: Sangyu Han, Yearim Kim, Nojun Kwak

**Abstract**: The truthfulness of existing explanation methods in authentically elucidating the underlying model's decision-making process has been questioned. Existing methods have deviated from faithfully representing the model, thus susceptible to adversarial attacks. To address this, we propose a novel eXplainable AI (XAI) method called SRD (Sharing Ratio Decomposition), which sincerely reflects the model's inference process, resulting in significantly enhanced robustness in our explanations. Different from the conventional emphasis on the neuronal level, we adopt a vector perspective to consider the intricate nonlinear interactions between filters. We also introduce an interesting observation termed Activation-Pattern-Only Prediction (APOP), letting us emphasize the importance of inactive neurons and redefine relevance encapsulating all relevant information including both active and inactive neurons. Our method, SRD, allows for the recursive decomposition of a Pointwise Feature Vector (PFV), providing a high-resolution Effective Receptive Field (ERF) at any layer.



## **9. Flexible Physical Camouflage Generation Based on a Differential Approach**

cs.CV

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2402.13575v3) [paper-pdf](http://arxiv.org/pdf/2402.13575v3)

**Authors**: Yang Li, Wenyi Tan, Tingrui Wang, Xinkai Liang, Quan Pan

**Abstract**: This study introduces a novel approach to neural rendering, specifically tailored for adversarial camouflage, within an extensive 3D rendering framework. Our method, named FPA, goes beyond traditional techniques by faithfully simulating lighting conditions and material variations, ensuring a nuanced and realistic representation of textures on a 3D target. To achieve this, we employ a generative approach that learns adversarial patterns from a diffusion model. This involves incorporating a specially designed adversarial loss and covert constraint loss to guarantee the adversarial and covert nature of the camouflage in the physical world. Furthermore, we showcase the effectiveness of the proposed camouflage in sticker mode, demonstrating its ability to cover the target without compromising adversarial information. Through empirical and physical experiments, FPA exhibits strong performance in terms of attack success rate and transferability. Additionally, the designed sticker-mode camouflage, coupled with a concealment constraint, adapts to the environment, yielding diverse styles of texture. Our findings highlight the versatility and efficacy of the FPA approach in adversarial camouflage applications.



## **10. AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization**

cs.CV

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2402.11940v4) [paper-pdf](http://arxiv.org/pdf/2402.11940v4)

**Authors**: Jiyao Li, Mingze Ni, Yifei Dong, Tianqing Zhu, Wei Liu

**Abstract**: Recent advances in deep learning research have shown remarkable achievements across many tasks in computer vision (CV) and natural language processing (NLP). At the intersection of CV and NLP is the problem of image captioning, where the related models' robustness against adversarial attacks has not been well studied. This paper presents a novel adversarial attack strategy, AICAttack (Attention-based Image Captioning Attack), designed to attack image captioning models through subtle perturbations on images. Operating within a black-box attack scenario, our algorithm requires no access to the target model's architecture, parameters, or gradient information. We introduce an attention-based candidate selection mechanism that identifies the optimal pixels to attack, followed by a customised differential evolution method to optimise the perturbations of pixels' RGB values. We demonstrate AICAttack's effectiveness through extensive experiments on benchmark datasets against multiple victim models. The experimental results demonstrate that our method outperforms current leading-edge techniques by achieving consistently higher attack success rates.



## **11. Exploiting the Index Gradients for Optimization-Based Jailbreaking on Large Language Models**

cs.CL

13 pages,2 figures, accepted by The 31st International Conference on  Computational Linguistics

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08615v1) [paper-pdf](http://arxiv.org/pdf/2412.08615v1)

**Authors**: Jiahui Li, Yongchang Hao, Haoyu Xu, Xing Wang, Yu Hong

**Abstract**: Despite the advancements in training Large Language Models (LLMs) with alignment techniques to enhance the safety of generated content, these models remain susceptible to jailbreak, an adversarial attack method that exposes security vulnerabilities in LLMs. Notably, the Greedy Coordinate Gradient (GCG) method has demonstrated the ability to automatically generate adversarial suffixes that jailbreak state-of-the-art LLMs. However, the optimization process involved in GCG is highly time-consuming, rendering the jailbreaking pipeline inefficient. In this paper, we investigate the process of GCG and identify an issue of Indirect Effect, the key bottleneck of the GCG optimization. To this end, we propose the Model Attack Gradient Index GCG (MAGIC), that addresses the Indirect Effect by exploiting the gradient information of the suffix tokens, thereby accelerating the procedure by having less computation and fewer iterations. Our experiments on AdvBench show that MAGIC achieves up to a 1.5x speedup, while maintaining Attack Success Rates (ASR) on par or even higher than other baselines. Our MAGIC achieved an ASR of 74% on the Llama-2 and an ASR of 54% when conducting transfer attacks on GPT-3.5. Code is available at https://github.com/jiah-li/magic.



## **12. AdvWave: Stealthy Adversarial Jailbreak Attack against Large Audio-Language Models**

cs.SD

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08608v1) [paper-pdf](http://arxiv.org/pdf/2412.08608v1)

**Authors**: Mintong Kang, Chejian Xu, Bo Li

**Abstract**: Recent advancements in large audio-language models (LALMs) have enabled speech-based user interactions, significantly enhancing user experience and accelerating the deployment of LALMs in real-world applications. However, ensuring the safety of LALMs is crucial to prevent risky outputs that may raise societal concerns or violate AI regulations. Despite the importance of this issue, research on jailbreaking LALMs remains limited due to their recent emergence and the additional technical challenges they present compared to attacks on DNN-based audio models. Specifically, the audio encoders in LALMs, which involve discretization operations, often lead to gradient shattering, hindering the effectiveness of attacks relying on gradient-based optimizations. The behavioral variability of LALMs further complicates the identification of effective (adversarial) optimization targets. Moreover, enforcing stealthiness constraints on adversarial audio waveforms introduces a reduced, non-convex feasible solution space, further intensifying the challenges of the optimization process. To overcome these challenges, we develop AdvWave, the first jailbreak framework against LALMs. We propose a dual-phase optimization method that addresses gradient shattering, enabling effective end-to-end gradient-based optimization. Additionally, we develop an adaptive adversarial target search algorithm that dynamically adjusts the adversarial optimization target based on the response patterns of LALMs for specific queries. To ensure that adversarial audio remains perceptually natural to human listeners, we design a classifier-guided optimization approach that generates adversarial noise resembling common urban sounds. Extensive evaluations on multiple advanced LALMs demonstrate that AdvWave outperforms baseline methods, achieving a 40% higher average jailbreak attack success rate.



## **13. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**

cs.CL

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2402.16822v3) [paper-pdf](http://arxiv.org/pdf/2402.16822v3)

**Authors**: Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu

**Abstract**: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to adversarial attacks is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel black-box approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem and uses open-ended search to generate prompts that are both effective and diverse. Focusing on the safety domain, we use Rainbow Teaming to target various state-of-the-art LLMs, including the Llama 2 and Llama 3 models. Our approach reveals hundreds of effective adversarial prompts, with an attack success rate exceeding 90% across all tested models. Furthermore, we demonstrate that prompts generated by Rainbow Teaming are highly transferable and that fine-tuning models with synthetic data generated by our method significantly enhances their safety without sacrificing general performance or helpfulness. We additionally explore the versatility of Rainbow Teaming by applying it to question answering and cybersecurity, showcasing its potential to drive robust open-ended self-improvement in a wide range of applications.



## **14. Grimm: A Plug-and-Play Perturbation Rectifier for Graph Neural Networks Defending against Poisoning Attacks**

cs.LG

19 pages, 13 figures

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08555v1) [paper-pdf](http://arxiv.org/pdf/2412.08555v1)

**Authors**: Ao Liu, Wenshan Li, Beibei Li, Wengang Ma, Tao Li, Pan Zhou

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.



## **15. Robust Deep Reinforcement Learning Through Adversarial Attacks and Training : A Survey**

cs.LG

61 pages, 17 figues, 1 table

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2403.00420v2) [paper-pdf](http://arxiv.org/pdf/2403.00420v2)

**Authors**: Lucas Schott, Josephine Delas, Hatem Hajri, Elies Gherbi, Reda Yaich, Nora Boulahia-Cuppens, Frederic Cuppens, Sylvain Lamprier

**Abstract**: Deep Reinforcement Learning (DRL) is a subfield of machine learning for training autonomous agents that take sequential actions across complex environments. Despite its significant performance in well-known environments, it remains susceptible to minor condition variations, raising concerns about its reliability in real-world applications. To improve usability, DRL must demonstrate trustworthiness and robustness. A way to improve the robustness of DRL to unknown changes in the environmental conditions and possible perturbations is through Adversarial Training, by training the agent against well-suited adversarial attacks on the observations and the dynamics of the environment. Addressing this critical issue, our work presents an in-depth analysis of contemporary adversarial attack and training methodologies, systematically categorizing them and comparing their objectives and operational mechanisms.



## **16. Adversarial Purification by Consistency-aware Latent Space Optimization on Data Manifolds**

cs.LG

17 pages, 8 figures

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08394v1) [paper-pdf](http://arxiv.org/pdf/2412.08394v1)

**Authors**: Shuhai Zhang, Jiahao Yang, Hui Luo, Jie Chen, Li Wang, Feng Liu, Bo Han, Mingkui Tan

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial samples crafted by adding imperceptible perturbations to clean data, potentially leading to incorrect and dangerous predictions. Adversarial purification has been an effective means to improve DNNs robustness by removing these perturbations before feeding the data into the model. However, it faces significant challenges in preserving key structural and semantic information of data, as the imperceptible nature of adversarial perturbations makes it hard to avoid over-correcting, which can destroy important information and degrade model performance. In this paper, we break away from traditional adversarial purification methods by focusing on the clean data manifold. To this end, we reveal that samples generated by a well-trained generative model are close to clean ones but far from adversarial ones. Leveraging this insight, we propose Consistency Model-based Adversarial Purification (CMAP), which optimizes vectors within the latent space of a pre-trained consistency model to generate samples for restoring clean data. Specifically, 1) we propose a \textit{Perceptual consistency restoration} mechanism by minimizing the discrepancy between generated samples and input samples in both pixel and perceptual spaces. 2) To maintain the optimized latent vectors within the valid data manifold, we introduce a \textit{Latent distribution consistency constraint} strategy to align generated samples with the clean data distribution. 3) We also apply a \textit{Latent vector consistency prediction} scheme via an ensemble approach to enhance prediction reliability. CMAP fundamentally addresses adversarial perturbations at their source, providing a robust purification. Extensive experiments on CIFAR-10 and ImageNet-100 show that our CMAP significantly enhances robustness against strong adversarial attacks while preserving high natural accuracy.



## **17. Graph Agent Network: Empowering Nodes with Inference Capabilities for Adversarial Resilience**

cs.LG

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2306.06909v5) [paper-pdf](http://arxiv.org/pdf/2306.06909v5)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Guangquan Xu, Pan Zhou, Wengang Ma, Hanyuan Huang

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.



## **18. How Does the Smoothness Approximation Method Facilitate Generalization for Federated Adversarial Learning?**

cs.LG

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08282v1) [paper-pdf](http://arxiv.org/pdf/2412.08282v1)

**Authors**: Wenjun Ding, Ying An, Lixing Chen, Shichao Kan, Fan Wu, Zhe Qu

**Abstract**: Federated Adversarial Learning (FAL) is a robust framework for resisting adversarial attacks on federated learning. Although some FAL studies have developed efficient algorithms, they primarily focus on convergence performance and overlook generalization. Generalization is crucial for evaluating algorithm performance on unseen data. However, generalization analysis is more challenging due to non-smooth adversarial loss functions. A common approach to addressing this issue is to leverage smoothness approximation. In this paper, we develop algorithm stability measures to evaluate the generalization performance of two popular FAL algorithms: \textit{Vanilla FAL (VFAL)} and {\it Slack FAL (SFAL)}, using three different smooth approximation methods: 1) \textit{Surrogate Smoothness Approximation (SSA)}, (2) \textit{Randomized Smoothness Approximation (RSA)}, and (3) \textit{Over-Parameterized Smoothness Approximation (OPSA)}. Based on our in-depth analysis, we answer the question of how to properly set the smoothness approximation method to mitigate generalization error in FAL. Moreover, we identify RSA as the most effective method for reducing generalization error. In highly data-heterogeneous scenarios, we also recommend employing SFAL to mitigate the deterioration of generalization performance caused by heterogeneity. Based on our theoretical results, we provide insights to help develop more efficient FAL algorithms, such as designing new metrics and dynamic aggregation rules to mitigate heterogeneity.



## **19. DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models**

cs.LG

Accepted by the Main Technical Track of the 39th Annual AAAI  Conference on Artificial Intelligence (AAAI-2025)

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.08160v2) [paper-pdf](http://arxiv.org/pdf/2412.08160v2)

**Authors**: Haonan Yuan, Qingyun Sun, Zhaonan Wang, Xingcheng Fu, Cheng Ji, Yongjian Wang, Bo Jin, Jianxin Li

**Abstract**: Dynamic graphs exhibit intertwined spatio-temporal evolutionary patterns, widely existing in the real world. Nevertheless, the structure incompleteness, noise, and redundancy result in poor robustness for Dynamic Graph Neural Networks (DGNNs). Dynamic Graph Structure Learning (DGSL) offers a promising way to optimize graph structures. However, aside from encountering unacceptable quadratic complexity, it overly relies on heuristic priors, making it hard to discover underlying predictive patterns. How to efficiently refine the dynamic structures, capture intrinsic dependencies, and learn robust representations, remains under-explored. In this work, we propose the novel DG-Mamba, a robust and efficient Dynamic Graph structure learning framework with the Selective State Space Models (Mamba). To accelerate the spatio-temporal structure learning, we propose a kernelized dynamic message-passing operator that reduces the quadratic time complexity to linear. To capture global intrinsic dynamics, we establish the dynamic graph as a self-contained system with State Space Model. By discretizing the system states with the cross-snapshot graph adjacency, we enable the long-distance dependencies capturing with the selective snapshot scan. To endow learned dynamic structures more expressive with informativeness, we propose the self-supervised Principle of Relevant Information for DGSL to regularize the most relevant yet least redundant information, enhancing global robustness. Extensive experiments demonstrate the superiority of the robustness and efficiency of our DG-Mamba compared with the state-of-the-art baselines against adversarial attacks.



## **20. Antelope: Potent and Concealed Jailbreak Attack Strategy**

cs.CR

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08156v1) [paper-pdf](http://arxiv.org/pdf/2412.08156v1)

**Authors**: Xin Zhao, Xiaojun Chen, Haoyu Gao

**Abstract**: Due to the remarkable generative potential of diffusion-based models, numerous researches have investigated jailbreak attacks targeting these frameworks. A particularly concerning threat within image models is the generation of Not-Safe-for-Work (NSFW) content. Despite the implementation of security filters, numerous efforts continue to explore ways to circumvent these safeguards. Current attack methodologies primarily encompass adversarial prompt engineering or concept obfuscation, yet they frequently suffer from slow search efficiency, conspicuous attack characteristics and poor alignment with targets. To overcome these challenges, we propose Antelope, a more robust and covert jailbreak attack strategy designed to expose security vulnerabilities inherent in generative models. Specifically, Antelope leverages the confusion of sensitive concepts with similar ones, facilitates searches in the semantically adjacent space of these related concepts and aligns them with the target imagery, thereby generating sensitive images that are consistent with the target and capable of evading detection. Besides, we successfully exploit the transferability of model-based attacks to penetrate online black-box services. Experimental evaluations demonstrate that Antelope outperforms existing baselines across multiple defensive mechanisms, underscoring its efficacy and versatility.



## **21. Doubly-Universal Adversarial Perturbations: Deceiving Vision-Language Models Across Both Images and Text with a Single Perturbation**

cs.CV

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08108v1) [paper-pdf](http://arxiv.org/pdf/2412.08108v1)

**Authors**: Hee-Seon Kim, Minbeom Kim, Changick Kim

**Abstract**: Large Vision-Language Models (VLMs) have demonstrated remarkable performance across multimodal tasks by integrating vision encoders with large language models (LLMs). However, these models remain vulnerable to adversarial attacks. Among such attacks, Universal Adversarial Perturbations (UAPs) are especially powerful, as a single optimized perturbation can mislead the model across various input images. In this work, we introduce a novel UAP specifically designed for VLMs: the Doubly-Universal Adversarial Perturbation (Doubly-UAP), capable of universally deceiving VLMs across both image and text inputs. To successfully disrupt the vision encoder's fundamental process, we analyze the core components of the attention mechanism. After identifying value vectors in the middle-to-late layers as the most vulnerable, we optimize Doubly-UAP in a label-free manner with a frozen model. Despite being developed as a black-box to the LLM, Doubly-UAP achieves high attack success rates on VLMs, consistently outperforming baseline methods across vision-language tasks. Extensive ablation studies and analyses further demonstrate the robustness of Doubly-UAP and provide insights into how it influences internal attention mechanisms.



## **22. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

cs.LG

11 pages, 5 figures

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08099v1) [paper-pdf](http://arxiv.org/pdf/2412.08099v1)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.



## **23. What You See Is Not Always What You Get: An Empirical Study of Code Comprehension by Large Language Models**

cs.SE

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08098v1) [paper-pdf](http://arxiv.org/pdf/2412.08098v1)

**Authors**: Bangshuo Zhu, Jiawen Wen, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering domain, covering numerous tasks such as code generation and comprehension. While the benefit of LLMs for coding task is well noted, it is perceived that LLMs are vulnerable to adversarial attacks. In this paper, we study the specific LLM vulnerability to imperceptible character attacks, a type of prompt-injection attack that uses special characters to befuddle an LLM whilst keeping the attack hidden to human eyes. We devise four categories of attacks and investigate their effects on the performance outcomes of tasks relating to code analysis and code comprehension. Two generations of ChatGPT are included to evaluate the impact of advancements made to contemporary models. Our experimental design consisted of comparing perturbed and unperturbed code snippets and evaluating two performance outcomes, which are model confidence using log probabilities of response, and correctness of response. We conclude that earlier version of ChatGPT exhibits a strong negative linear correlation between the amount of perturbation and the performance outcomes, while the recent ChatGPT presents a strong negative correlation between the presence of perturbation and performance outcomes, but no valid correlational relationship between perturbation budget and performance outcomes. We anticipate this work contributes to an in-depth understanding of leveraging LLMs for coding tasks. It is suggested future research should delve into how to create LLMs that can return a correct response even if the prompt exhibits perturbations.



## **24. Plentiful Jailbreaks with String Compositions**

cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2411.01084v3) [paper-pdf](http://arxiv.org/pdf/2411.01084v3)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or red-teamers, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary string compositions, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.



## **25. DynamicPAE: Generating Scene-Aware Physical Adversarial Examples in Real-Time**

cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08053v1) [paper-pdf](http://arxiv.org/pdf/2412.08053v1)

**Authors**: Jin Hu, Xianglong Liu, Jiakai Wang, Junkai Zhang, Xianqi Yang, Haotong Qin, Yuqing Ma, Ke Xu

**Abstract**: Physical adversarial examples (PAEs) are regarded as "whistle-blowers" of real-world risks in deep-learning applications. However, current PAE generation studies show limited adaptive attacking ability to diverse and varying scenes. The key challenges in generating dynamic PAEs are exploring their patterns under noisy gradient feedback and adapting the attack to agnostic scenario natures. To address the problems, we present DynamicPAE, the first generative framework that enables scene-aware real-time physical attacks beyond static attacks. Specifically, to train the dynamic PAE generator under noisy gradient feedback, we introduce the residual-driven sample trajectory guidance technique, which redefines the training task to break the limited feedback information restriction that leads to the degeneracy problem. Intuitively, it allows the gradient feedback to be passed to the generator through a low-noise auxiliary task, thereby guiding the optimization away from degenerate solutions and facilitating a more comprehensive and stable exploration of feasible PAEs. To adapt the generator to agnostic scenario natures, we introduce the context-aligned scene expectation simulation process, consisting of the conditional-uncertainty-aligned data module and the skewness-aligned objective re-weighting module. The former enhances robustness in the context of incomplete observation by employing a conditional probabilistic model for domain randomization, while the latter facilitates consistent stealth control across different attack targets by automatically reweighting losses based on the skewness indicator. Extensive digital and physical evaluations demonstrate the superior attack performance of DynamicPAE, attaining a 1.95 $\times$ boost (65.55% average AP drop under attack) on representative object detectors (e.g., Yolo-v8) over state-of-the-art static PAE generating methods.



## **26. GLL: A Differentiable Graph Learning Layer for Neural Networks**

cs.LG

44 pages, 11 figures. Preprint. Submitted to the Journal of Machine  Learning Research

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08016v1) [paper-pdf](http://arxiv.org/pdf/2412.08016v1)

**Authors**: Jason Brown, Bohan Chen, Harris Hardiman-Mostow, Jeff Calder, Andrea L. Bertozzi

**Abstract**: Standard deep learning architectures used for classification generate label predictions with a projection head and softmax activation function. Although successful, these methods fail to leverage the relational information between samples in the batch for generating label predictions. In recent works, graph-based learning techniques, namely Laplace learning, have been heuristically combined with neural networks for both supervised and semi-supervised learning (SSL) tasks. However, prior works approximate the gradient of the loss function with respect to the graph learning algorithm or decouple the processes; end-to-end integration with neural networks is not achieved. In this work, we derive backpropagation equations, via the adjoint method, for inclusion of a general family of graph learning layers into a neural network. This allows us to precisely integrate graph Laplacian-based label propagation into a neural network layer, replacing a projection head and softmax activation function for classification tasks. Using this new framework, our experimental results demonstrate smooth label transitions across data, improved robustness to adversarial attacks, improved generalization, and improved training dynamics compared to the standard softmax-based approach.



## **27. MAGIC: Mastering Physical Adversarial Generation in Context through Collaborative LLM Agents**

cs.CV

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08014v1) [paper-pdf](http://arxiv.org/pdf/2412.08014v1)

**Authors**: Yun Xing, Nhat Chung, Jie Zhang, Yue Cao, Ivor Tsang, Yang Liu, Lei Ma, Qing Guo

**Abstract**: Physical adversarial attacks in driving scenarios can expose critical vulnerabilities in visual perception models. However, developing such attacks remains challenging due to diverse real-world backgrounds and the requirement for maintaining visual naturality. Building upon this challenge, we reformulate physical adversarial attacks as a one-shot patch-generation problem. Our approach generates adversarial patches through a deep generative model that considers the specific scene context, enabling direct physical deployment in matching environments. The primary challenge lies in simultaneously achieving two objectives: generating adversarial patches that effectively mislead object detection systems while determining contextually appropriate placement within the scene. We propose MAGIC (Mastering Physical Adversarial Generation In Context), a novel framework powered by multi-modal LLM agents to address these challenges. MAGIC automatically understands scene context and orchestrates adversarial patch generation through the synergistic interaction of language and vision capabilities. MAGIC orchestrates three specialized LLM agents: The adv-patch generation agent (GAgent) masters the creation of deceptive patches through strategic prompt engineering for text-to-image models. The adv-patch deployment agent (DAgent) ensures contextual coherence by determining optimal placement strategies based on scene understanding. The self-examination agent (EAgent) completes this trilogy by providing critical oversight and iterative refinement of both processes. We validate our method on both digital and physical level, \ie, nuImage and manually captured real scenes, where both statistical and visual results prove that our MAGIC is powerful and effectively for attacking wide-used object detection systems.



## **28. Enhancing Remote Adversarial Patch Attacks on Face Detectors with Tiling and Scaling**

cs.CV

Accepted and Presented at APSIPA ASC 2024

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.07996v1) [paper-pdf](http://arxiv.org/pdf/2412.07996v1)

**Authors**: Masora Okano, Koichi Ito, Masakatsu Nishigaki, Tetsushi Ohki

**Abstract**: This paper discusses the attack feasibility of Remote Adversarial Patch (RAP) targeting face detectors. The RAP that targets face detectors is similar to the RAP that targets general object detectors, but the former has multiple issues in the attack process the latter does not. (1) It is possible to detect objects of various scales. In particular, the area of small objects that are convolved during feature extraction by CNN is small,so the area that affects the inference results is also small. (2) It is a two-class classification, so there is a large gap in characteristics between the classes. This makes it difficult to attack the inference results by directing them to a different class. In this paper, we propose a new patch placement method and loss function for each problem. The patches targeting the proposed face detector showed superior detection obstruct effects compared to the patches targeting the general object detector.



## **29. PBP: Post-training Backdoor Purification for Malware Classifiers**

cs.LG

Accepted at NDSS 2025

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.03441v3) [paper-pdf](http://arxiv.org/pdf/2412.03441v3)

**Authors**: Dung Thuy Nguyen, Ngoc N. Tran, Taylor T. Johnson, Kevin Leach

**Abstract**: In recent years, the rise of machine learning (ML) in cybersecurity has brought new challenges, including the increasing threat of backdoor poisoning attacks on ML malware classifiers. For instance, adversaries could inject malicious samples into public malware repositories, contaminating the training data and potentially misclassifying malware by the ML model. Current countermeasures predominantly focus on detecting poisoned samples by leveraging disagreements within the outputs of a diverse set of ensemble models on training data points. However, these methods are not suitable for scenarios where Machine Learning-as-a-Service (MLaaS) is used or when users aim to remove backdoors from a model after it has been trained. Addressing this scenario, we introduce PBP, a post-training defense for malware classifiers that mitigates various types of backdoor embeddings without assuming any specific backdoor embedding mechanism. Our method exploits the influence of backdoor attacks on the activation distribution of neural networks, independent of the trigger-embedding method. In the presence of a backdoor attack, the activation distribution of each layer is distorted into a mixture of distributions. By regulating the statistics of the batch normalization layers, we can guide a backdoored model to perform similarly to a clean one. Our method demonstrates substantial advantages over several state-of-the-art methods, as evidenced by experiments on two datasets, two types of backdoor methods, and various attack configurations. Notably, our approach requires only a small portion of the training data -- only 1\% -- to purify the backdoor and reduce the attack success rate from 100\% to almost 0\%, a 100-fold improvement over the baseline methods. Our code is available at \url{https://github.com/judydnguyen/pbp-backdoor-purification-official}.



## **30. DeMem: Privacy-Enhanced Robust Adversarial Learning via De-Memorization**

cs.LG

10 pages

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.05767v2) [paper-pdf](http://arxiv.org/pdf/2412.05767v2)

**Authors**: Xiaoyu Luo, Qiongxiu Li

**Abstract**: Adversarial robustness, the ability of a model to withstand manipulated inputs that cause errors, is essential for ensuring the trustworthiness of machine learning models in real-world applications. However, previous studies have shown that enhancing adversarial robustness through adversarial training increases vulnerability to privacy attacks. While differential privacy can mitigate these attacks, it often compromises robustness against both natural and adversarial samples. Our analysis reveals that differential privacy disproportionately impacts low-risk samples, causing an unintended performance drop. To address this, we propose DeMem, which selectively targets high-risk samples, achieving a better balance between privacy protection and model robustness. DeMem is versatile and can be seamlessly integrated into various adversarial training techniques. Extensive evaluations across multiple training methods and datasets demonstrate that DeMem significantly reduces privacy leakage while maintaining robustness against both natural and adversarial samples. These results confirm DeMem's effectiveness and broad applicability in enhancing privacy without compromising robustness.



## **31. Defending Against Neural Network Model Inversion Attacks via Data Poisoning**

cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07575v1) [paper-pdf](http://arxiv.org/pdf/2412.07575v1)

**Authors**: Shuai Zhou, Dayong Ye, Tianqing Zhu, Wanlei Zhou

**Abstract**: Model inversion attacks pose a significant privacy threat to machine learning models by reconstructing sensitive data from their outputs. While various defenses have been proposed to counteract these attacks, they often come at the cost of the classifier's utility, thus creating a challenging trade-off between privacy protection and model utility. Moreover, most existing defenses require retraining the classifier for enhanced robustness, which is impractical for large-scale, well-established models. This paper introduces a novel defense mechanism to better balance privacy and utility, particularly against adversaries who employ a machine learning model (i.e., inversion model) to reconstruct private data. Drawing inspiration from data poisoning attacks, which can compromise the performance of machine learning models, we propose a strategy that leverages data poisoning to contaminate the training data of inversion models, thereby preventing model inversion attacks.   Two defense methods are presented. The first, termed label-preserving poisoning attacks for all output vectors (LPA), involves subtle perturbations to all output vectors while preserving their labels. Our findings demonstrate that these minor perturbations, introduced through a data poisoning approach, significantly increase the difficulty of data reconstruction without compromising the utility of the classifier. Subsequently, we introduce a second method, label-flipping poisoning for partial output vectors (LFP), which selectively perturbs a small subset of output vectors and alters their labels during the process. Empirical results indicate that LPA is notably effective, outperforming the current state-of-the-art defenses. Our data poisoning-based defense provides a new retraining-free defense paradigm that preserves the victim classifier's utility.



## **32. Adaptive Epsilon Adversarial Training for Robust Gravitational Wave Parameter Estimation Using Normalizing Flows**

cs.LG

7 pages, 9 figures

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07559v1) [paper-pdf](http://arxiv.org/pdf/2412.07559v1)

**Authors**: Yiqian Yang, Xihua Zhu, Fan Zhang

**Abstract**: Adversarial training with Normalizing Flow (NF) models is an emerging research area aimed at improving model robustness through adversarial samples. In this study, we focus on applying adversarial training to NF models for gravitational wave parameter estimation. We propose an adaptive epsilon method for Fast Gradient Sign Method (FGSM) adversarial training, which dynamically adjusts perturbation strengths based on gradient magnitudes using logarithmic scaling. Our hybrid architecture, combining ResNet and Inverse Autoregressive Flow, reduces the Negative Log Likelihood (NLL) loss by 47\% under FGSM attacks compared to the baseline model, while maintaining an NLL of 4.2 on clean data (only 5\% higher than the baseline). For perturbation strengths between 0.01 and 0.1, our model achieves an average NLL of 5.8, outperforming both fixed-epsilon (NLL: 6.7) and progressive-epsilon (NLL: 7.2) methods. Under stronger Projected Gradient Descent attacks with perturbation strength of 0.05, our model maintains an NLL of 6.4, demonstrating superior robustness while avoiding catastrophic overfitting.



## **33. Quantifying the Prediction Uncertainty of Machine Learning Models for Individual Data**

cs.LG

PHD thesis

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07520v1) [paper-pdf](http://arxiv.org/pdf/2412.07520v1)

**Authors**: Koby Bibas

**Abstract**: Machine learning models have exhibited exceptional results in various domains. The most prevalent approach for learning is the empirical risk minimizer (ERM), which adapts the model's weights to reduce the loss on a training set and subsequently leverages these weights to predict the label for new test data. Nonetheless, ERM makes the assumption that the test distribution is similar to the training distribution, which may not always hold in real-world situations. In contrast, the predictive normalized maximum likelihood (pNML) was proposed as a min-max solution for the individual setting where no assumptions are made on the distribution of the tested input. This study investigates pNML's learnability for linear regression and neural networks, and demonstrates that pNML can improve the performance and robustness of these models on various tasks. Moreover, the pNML provides an accurate confidence measure for its output, showcasing state-of-the-art results for out-of-distribution detection, resistance to adversarial attacks, and active learning.



## **34. AHSG: Adversarial Attacks on High-level Semantics in Graph Neural Networks**

cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07468v1) [paper-pdf](http://arxiv.org/pdf/2412.07468v1)

**Authors**: Kai Yuan, Xiaobing Pei, Haoran Yang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant interest among researchers due to their impressive performance in graph learning tasks. However, like other deep neural networks, GNNs are also vulnerable to adversarial attacks. In existing adversarial attack methods for GNNs, the metric between the attacked graph and the original graph is usually the attack budget or a measure of global graph properties. However, we have found that it is possible to generate attack graphs that disrupt the primary semantics even within these constraints. To address this problem, we propose a Adversarial Attacks on High-level Semantics in Graph Neural Networks (AHSG), which is a graph structure attack model that ensures the retention of primary semantics. The latent representations of each node can extract rich semantic information by applying convolutional operations on graph data. These representations contain both task-relevant primary semantic information and task-irrelevant secondary semantic information. The latent representations of same-class nodes with the same primary semantics can fulfill the objective of modifying secondary semantics while preserving the primary semantics. Finally, the latent representations with attack effects is mapped to an attack graph using Projected Gradient Descent (PGD) algorithm. By attacking graph deep learning models with some advanced defense strategies, we validate that AHSG has superior attack effectiveness compared to other attack methods. Additionally, we employ Contextual Stochastic Block Models (CSBMs) as a proxy for the primary semantics to detect the attacked graph, confirming that AHSG almost does not disrupt the original primary semantics of the graph.



## **35. Addressing Key Challenges of Adversarial Attacks and Defenses in the Tabular Domain: A Methodological Framework for Coherence and Consistency**

cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07326v1) [paper-pdf](http://arxiv.org/pdf/2412.07326v1)

**Authors**: Yael Itzhakev, Amit Giloni, Yuval Elovici, Asaf Shabtai

**Abstract**: Machine learning models trained on tabular data are vulnerable to adversarial attacks, even in realistic scenarios where attackers have access only to the model's outputs. Researchers evaluate such attacks by considering metrics like success rate, perturbation magnitude, and query count. However, unlike other data domains, the tabular domain contains complex interdependencies among features, presenting a unique aspect that should be evaluated: the need for the attack to generate coherent samples and ensure feature consistency for indistinguishability. Currently, there is no established methodology for evaluating adversarial samples based on these criteria. In this paper, we address this gap by proposing new evaluation criteria tailored for tabular attacks' quality; we defined anomaly-based framework to assess the distinguishability of adversarial samples and utilize the SHAP explainability technique to identify inconsistencies in the model's decision-making process caused by adversarial samples. These criteria could form the basis for potential detection methods and be integrated into established evaluation metrics for assessing attack's quality Additionally, we introduce a novel technique for perturbing dependent features while maintaining coherence and feature consistency within the sample. We compare different attacks' strategies, examining black-box query-based attacks and transferability-based gradient attacks across four target models. Our experiments, conducted on benchmark tabular datasets, reveal significant differences between the examined attacks' strategies in terms of the attacker's risk and effort and the attacks' quality. The findings provide valuable insights on the strengths, limitations, and trade-offs of various adversarial attacks in the tabular domain, laying a foundation for future research on attacks and defense development.



## **36. Backdoor Attacks against No-Reference Image Quality Assessment Models via A Scalable Trigger**

cs.CV

Accept by AAAI 2025

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07277v1) [paper-pdf](http://arxiv.org/pdf/2412.07277v1)

**Authors**: Yi Yu, Song Xia, Xun Lin, Wenhan Yang, Shijian Lu, Yap-peng Tan, Alex Kot

**Abstract**: No-Reference Image Quality Assessment (NR-IQA), responsible for assessing the quality of a single input image without using any reference, plays a critical role in evaluating and optimizing computer vision systems, e.g., low-light enhancement. Recent research indicates that NR-IQA models are susceptible to adversarial attacks, which can significantly alter predicted scores with visually imperceptible perturbations. Despite revealing vulnerabilities, these attack methods have limitations, including high computational demands, untargeted manipulation, limited practical utility in white-box scenarios, and reduced effectiveness in black-box scenarios. To address these challenges, we shift our focus to another significant threat and present a novel poisoning-based backdoor attack against NR-IQA (BAIQA), allowing the attacker to manipulate the IQA model's output to any desired target value by simply adjusting a scaling coefficient $\alpha$ for the trigger. We propose to inject the trigger in the discrete cosine transform (DCT) domain to improve the local invariance of the trigger for countering trigger diminishment in NR-IQA models due to widely adopted data augmentations. Furthermore, the universal adversarial perturbations (UAP) in the DCT space are designed as the trigger, to increase IQA model susceptibility to manipulation and improve attack effectiveness. In addition to the heuristic method for poison-label BAIQA (P-BAIQA), we explore the design of clean-label BAIQA (C-BAIQA), focusing on $\alpha$ sampling and image data refinement, driven by theoretical insights we reveal. Extensive experiments on diverse datasets and various NR-IQA models demonstrate the effectiveness of our attacks. Code will be released at https://github.com/yuyi-sd/BAIQA.



## **37. A Generative Victim Model for Segmentation**

cs.CV

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07274v1) [paper-pdf](http://arxiv.org/pdf/2412.07274v1)

**Authors**: Aixuan Li, Jing Zhang, Jiawei Shi, Yiran Zhong, Yuchao Dai

**Abstract**: We find that the well-trained victim models (VMs), against which the attacks are generated, serve as fundamental prerequisites for adversarial attacks, i.e. a segmentation VM is needed to generate attacks for segmentation. In this context, the victim model is assumed to be robust to achieve effective adversarial perturbation generation. Instead of focusing on improving the robustness of the task-specific victim models, we shift our attention to image generation. From an image generation perspective, we derive a novel VM for segmentation, aiming to generate adversarial perturbations for segmentation tasks without requiring models explicitly designed for image segmentation. Our approach to adversarial attack generation diverges from conventional white-box or black-box attacks, offering a fresh outlook on adversarial attack strategies. Experiments show that our attack method is able to generate effective adversarial attacks with good transferability.



## **38. CapGen:An Environment-Adaptive Generator of Adversarial Patches**

cs.CV

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07253v1) [paper-pdf](http://arxiv.org/pdf/2412.07253v1)

**Authors**: Chaoqun Li, Zhuodong Liu, Huanqian Yan, Hang Su

**Abstract**: Adversarial patches, often used to provide physical stealth protection for critical assets and assess perception algorithm robustness, usually neglect the need for visual harmony with the background environment, making them easily noticeable. Moreover, existing methods primarily concentrate on improving attack performance, disregarding the intricate dynamics of adversarial patch elements. In this work, we introduce the Camouflaged Adversarial Pattern Generator (CAPGen), a novel approach that leverages specific base colors from the surrounding environment to produce patches that seamlessly blend with their background for superior visual stealthiness while maintaining robust adversarial performance. We delve into the influence of both patterns (i.e., color-agnostic texture information) and colors on the effectiveness of attacks facilitated by patches, discovering that patterns exert a more pronounced effect on performance than colors. Based on these findings, we propose a rapid generation strategy for adversarial patches. This involves updating the colors of high-performance adversarial patches to align with those of the new environment, ensuring visual stealthiness without compromising adversarial impact. This paper is the first to comprehensively examine the roles played by patterns and colors in the context of adversarial patches.



## **39. Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces**

cs.HC

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07231v1) [paper-pdf](http://arxiv.org/pdf/2412.07231v1)

**Authors**: Lubin Meng, Xue Jiang, Xiaoqing Chen, Wenzhong Liu, Hanbin Luo, Dongrui Wu

**Abstract**: A brain-computer interface (BCI) enables direct communication between the brain and an external device. Electroencephalogram (EEG) is a common input signal for BCIs, due to its convenience and low cost. Most research on EEG-based BCIs focuses on the accurate decoding of EEG signals, while ignoring their security. Recent studies have shown that machine learning models in BCIs are vulnerable to adversarial attacks. This paper proposes adversarial filtering based evasion and backdoor attacks to EEG-based BCIs, which are very easy to implement. Experiments on three datasets from different BCI paradigms demonstrated the effectiveness of our proposed attack approaches. To our knowledge, this is the first study on adversarial filtering for EEG-based BCIs, raising a new security concern and calling for more attention on the security of BCIs.



## **40. A Parametric Approach to Adversarial Augmentation for Cross-Domain Iris Presentation Attack Detection**

cs.CV

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV),  2025

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07199v1) [paper-pdf](http://arxiv.org/pdf/2412.07199v1)

**Authors**: Debasmita Pal, Redwan Sony, Arun Ross

**Abstract**: Iris-based biometric systems are vulnerable to presentation attacks (PAs), where adversaries present physical artifacts (e.g., printed iris images, textured contact lenses) to defeat the system. This has led to the development of various presentation attack detection (PAD) algorithms, which typically perform well in intra-domain settings. However, they often struggle to generalize effectively in cross-domain scenarios, where training and testing employ different sensors, PA instruments, and datasets. In this work, we use adversarial training samples of both bonafide irides and PAs to improve the cross-domain performance of a PAD classifier. The novelty of our approach lies in leveraging transformation parameters from classical data augmentation schemes (e.g., translation, rotation) to generate adversarial samples. We achieve this through a convolutional autoencoder, ADV-GEN, that inputs original training samples along with a set of geometric and photometric transformations. The transformation parameters act as regularization variables, guiding ADV-GEN to generate adversarial samples in a constrained search space. Experiments conducted on the LivDet-Iris 2017 database, comprising four datasets, and the LivDet-Iris 2020 dataset, demonstrate the efficacy of our proposed method. The code is available at https://github.com/iPRoBe-lab/ADV-GEN-IrisPAD.



## **41. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07192v1) [paper-pdf](http://arxiv.org/pdf/2412.07192v1)

**Authors**: Zachary Coalson, Jeonghyun Woo, Shiyang Chen, Yu Sun, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.



## **42. dSTAR: Straggler Tolerant and Byzantine Resilient Distributed SGD**

cs.DC

15 pages

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07151v1) [paper-pdf](http://arxiv.org/pdf/2412.07151v1)

**Authors**: Jiahe Yan, Pratik Chaudhari, Leonard Kleinrock

**Abstract**: Distributed model training needs to be adapted to challenges such as the straggler effect and Byzantine attacks. When coordinating the training process with multiple computing nodes, ensuring timely and reliable gradient aggregation amidst network and system malfunctions is essential. To tackle these issues, we propose \textit{dSTAR}, a lightweight and efficient approach for distributed stochastic gradient descent (SGD) that enhances robustness and convergence. \textit{dSTAR} selectively aggregates gradients by collecting updates from the first \(k\) workers to respond, filtering them based on deviations calculated using an ensemble median. This method not only mitigates the impact of stragglers but also fortifies the model against Byzantine adversaries. We theoretically establish that \textit{dSTAR} is (\(\alpha, f\))-Byzantine resilient and achieves a linear convergence rate. Empirical evaluations across various scenarios demonstrate that \textit{dSTAR} consistently maintains high accuracy, outperforming other Byzantine-resilient methods that often suffer up to a 40-50\% accuracy drop under attack. Our results highlight \textit{dSTAR} as a robust solution for training models in distributed environments prone to both straggler delays and Byzantine faults.



## **43. Defensive Dual Masking for Robust Adversarial Defense**

cs.CL

First version

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07078v1) [paper-pdf](http://arxiv.org/pdf/2412.07078v1)

**Authors**: Wangli Yang, Jie Yang, Yi Guo, Johan Barthelemy

**Abstract**: The field of textual adversarial defenses has gained considerable attention in recent years due to the increasing vulnerability of natural language processing (NLP) models to adversarial attacks, which exploit subtle perturbations in input text to deceive models. This paper introduces the Defensive Dual Masking (DDM) algorithm, a novel approach designed to enhance model robustness against such attacks. DDM utilizes a unique adversarial training strategy where [MASK] tokens are strategically inserted into training samples to prepare the model to handle adversarial perturbations more effectively. During inference, potentially adversarial tokens are dynamically replaced with [MASK] tokens to neutralize potential threats while preserving the core semantics of the input. The theoretical foundation of our approach is explored, demonstrating how the selective masking mechanism strengthens the model's ability to identify and mitigate adversarial manipulations. Our empirical evaluation across a diverse set of benchmark datasets and attack mechanisms consistently shows that DDM outperforms state-of-the-art defense techniques, improving model accuracy and robustness. Moreover, when applied to Large Language Models (LLMs), DDM also enhances their resilience to adversarial attacks, providing a scalable defense mechanism for large-scale NLP applications.



## **44. Dense Cross-Connected Ensemble Convolutional Neural Networks for Enhanced Model Robustness**

cs.CV

6 pages, 1 figure

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.07022v1) [paper-pdf](http://arxiv.org/pdf/2412.07022v1)

**Authors**: Longwei Wang, Xueqian Li, Zheng Zhang

**Abstract**: The resilience of convolutional neural networks against input variations and adversarial attacks remains a significant challenge in image recognition tasks. Motivated by the need for more robust and reliable image recognition systems, we propose the Dense Cross-Connected Ensemble Convolutional Neural Network (DCC-ECNN). This novel architecture integrates the dense connectivity principle of DenseNet with the ensemble learning strategy, incorporating intermediate cross-connections between different DenseNet paths to facilitate extensive feature sharing and integration. The DCC-ECNN architecture leverages DenseNet's efficient parameter usage and depth while benefiting from the robustness of ensemble learning, ensuring a richer and more resilient feature representation.



## **45. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

quant-ph

58 pages, 4 figures; accepted in Quantum

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2204.02265v5) [paper-pdf](http://arxiv.org/pdf/2204.02265v5)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstract**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$-bit output to have some randomness when conditioned on the $n$-bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a fully black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQS model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the fully-black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC 2013) to the CRQS model. Second, we show a fully-black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt 2019) where quantum bolts have an additional parameter that cannot be changed without generating new bolts. Our results also apply to $2$-message protocols in the plain model.



## **46. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

cs.CL

NeurIPS 2024 Camera Ready. First two authors contributed equally.  Third and fourth authors contributed equally

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2406.18495v3) [paper-pdf](http://arxiv.org/pdf/2406.18495v3)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.



## **47. Take Fake as Real: Realistic-like Robust Black-box Adversarial Attack to Evade AIGC Detection**

cs.CV

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06727v1) [paper-pdf](http://arxiv.org/pdf/2412.06727v1)

**Authors**: Caiyun Xie, Dengpan Ye, Yunming Zhang, Long Tang, Yunna Lv, Jiacheng Deng, Jiawei Song

**Abstract**: The security of AI-generated content (AIGC) detection based on GANs and diffusion models is closely related to the credibility of multimedia content. Malicious adversarial attacks can evade these developing AIGC detection. However, most existing adversarial attacks focus only on GAN-generated facial images detection, struggle to be effective on multi-class natural images and diffusion-based detectors, and exhibit poor invisibility. To fill this gap, we first conduct an in-depth analysis of the vulnerability of AIGC detectors and discover the feature that detectors vary in vulnerability to different post-processing. Then, considering the uncertainty of detectors in real-world scenarios, and based on the discovery, we propose a Realistic-like Robust Black-box Adversarial attack (R$^2$BA) with post-processing fusion optimization. Unlike typical perturbations, R$^2$BA uses real-world post-processing, i.e., Gaussian blur, JPEG compression, Gaussian noise and light spot to generate adversarial examples. Specifically, we use a stochastic particle swarm algorithm with inertia decay to optimize post-processing fusion intensity and explore the detector's decision boundary. Guided by the detector's fake probability, R$^2$BA enhances/weakens the detector-vulnerable/detector-robust post-processing intensity to strike a balance between adversariality and invisibility. Extensive experiments on popular/commercial AIGC detectors and datasets demonstrate that R$^2$BA exhibits impressive anti-detection performance, excellent invisibility, and strong robustness in GAN-based and diffusion-based cases. Compared to state-of-the-art white-box and black-box attacks, R$^2$BA shows significant improvements of 15% and 21% in anti-detection performance under the original and robust scenario respectively, offering valuable insights for the security of AIGC detection in real-world applications.



## **48. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

cs.CR

15 pages, 13 figures

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2202.03195v6) [paper-pdf](http://arxiv.org/pdf/2202.03195v6)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against one defense. We find that both attacks are robust against the investigated defense, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.



## **49. Vulnerability, Where Art Thou? An Investigation of Vulnerability Management in Android Smartphone Chipsets**

cs.CR

Accepted by Network and Distributed System Security (NDSS) Symposium  2025

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06556v1) [paper-pdf](http://arxiv.org/pdf/2412.06556v1)

**Authors**: Daniel Klischies, Philipp Mackensen, Veelasha Moonsamy

**Abstract**: Vulnerabilities in Android smartphone chipsets have severe consequences, as recent real-world attacks have demonstrated that adversaries can leverage vulnerabilities to execute arbitrary code or exfiltrate confidential information. Despite the far-reaching impact of such attacks, the lifecycle of chipset vulnerabilities has yet to be investigated, with existing papers primarily investigating vulnerabilities in the Android operating system. This paper provides a comprehensive and empirical study of the current state of smartphone chipset vulnerability management within the Android ecosystem. For the first time, we create a unified knowledge base of 3,676 chipset vulnerabilities affecting 437 chipset models from all four major chipset manufacturers, combined with 6,866 smartphone models. Our analysis revealed that the same vulnerabilities are often included in multiple generations of chipsets, providing novel empirical evidence that vulnerabilities are inherited through multiple chipset generations. Furthermore, we demonstrate that the commonly accepted 90-day responsible vulnerability disclosure period is seldom adhered to. We find that a single vulnerability often affects hundreds to thousands of different smartphone models, for which update availability is, as we show, often unclear or heavily delayed. Leveraging the new insights gained from our empirical analysis, we recommend several changes that chipset manufacturers can implement to improve the security posture of their products. At the same time, our knowledge base enables academic researchers to conduct more representative evaluations of smartphone chipsets, accurately assess the impact of vulnerabilities they discover, and identify avenues for future research.



## **50. Flexible and Scalable Deep Dendritic Spiking Neural Networks with Multiple Nonlinear Branching**

cs.NE

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06355v1) [paper-pdf](http://arxiv.org/pdf/2412.06355v1)

**Authors**: Yifan Huang, Wei Fang, Zhengyu Ma, Guoqi Li, Yonghong Tian

**Abstract**: Recent advances in spiking neural networks (SNNs) have a predominant focus on network architectures, while relatively little attention has been paid to the underlying neuron model. The point neuron models, a cornerstone of deep SNNs, pose a bottleneck on the network-level expressivity since they depict somatic dynamics only. In contrast, the multi-compartment models in neuroscience offer remarkable expressivity by introducing dendritic morphology and dynamics, but remain underexplored in deep learning due to their unaffordable computational cost and inflexibility. To combine the advantages of both sides for a flexible, efficient yet more powerful model, we propose the dendritic spiking neuron (DendSN) incorporating multiple dendritic branches with nonlinear dynamics. Compared to the point spiking neurons, DendSN exhibits significantly higher expressivity. DendSN's flexibility enables its seamless integration into diverse deep SNN architectures. To accelerate dendritic SNNs (DendSNNs), we parallelize dendritic state updates across time steps, and develop Triton kernels for GPU-level acceleration. As a result, we can construct large-scale DendSNNs with depth comparable to their point SNN counterparts. Next, we comprehensively evaluate DendSNNs' performance on various demanding tasks. By modulating dendritic branch strengths using a context signal, catastrophic forgetting of DendSNNs is substantially mitigated. Moreover, DendSNNs demonstrate enhanced robustness against noise and adversarial attacks compared to point SNNs, and excel in few-shot learning settings. Our work firstly demonstrates the possibility of training bio-plausible dendritic SNNs with depths and scales comparable to traditional point SNNs, and reveals superior expressivity and robustness of reduced dendritic neuron models in deep learning, thereby offering a fresh perspective on advancing neural network design.



