# Latest Adversarial Attack Papers
**update at 2024-09-14 09:33:52**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. LoRID: Low-Rank Iterative Diffusion for Adversarial Purification**

cs.LG

LA-UR-24-28834

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08255v1) [paper-pdf](http://arxiv.org/pdf/2409.08255v1)

**Authors**: Geigh Zollicoffer, Minh Vu, Ben Nebgen, Juan Castorena, Boian Alexandrov, Manish Bhattarai

**Abstract**: This work presents an information-theoretic examination of diffusion-based purification methods, the state-of-the-art adversarial defenses that utilize diffusion models to remove malicious perturbations in adversarial examples. By theoretically characterizing the inherent purification errors associated with the Markov-based diffusion purifications, we introduce LoRID, a novel Low-Rank Iterative Diffusion purification method designed to remove adversarial perturbation with low intrinsic purification errors. LoRID centers around a multi-stage purification process that leverages multiple rounds of diffusion-denoising loops at the early time-steps of the diffusion models, and the integration of Tucker decomposition, an extension of matrix factorization, to remove adversarial noise at high-noise regimes. Consequently, LoRID increases the effective diffusion time-steps and overcomes strong adversarial attacks, achieving superior robustness performance in CIFAR-10/100, CelebA-HQ, and ImageNet datasets under both white-box and black-box settings.



## **2. High-Frequency Anti-DreamBooth: Robust Defense Against Image Synthesis**

cs.CV

ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08167v1) [paper-pdf](http://arxiv.org/pdf/2409.08167v1)

**Authors**: Takuto Onikubo, Yusuke Matsui

**Abstract**: Recently, text-to-image generative models have been misused to create unauthorized malicious images of individuals, posing a growing social problem. Previous solutions, such as Anti-DreamBooth, add adversarial noise to images to protect them from being used as training data for malicious generation. However, we found that the adversarial noise can be removed by adversarial purification methods such as DiffPure. Therefore, we propose a new adversarial attack method that adds strong perturbation on the high-frequency areas of images to make it more robust to adversarial purification. Our experiment showed that the adversarial images retained noise even after adversarial purification, hindering malicious image generation.



## **3. Unleashing Worms and Extracting Data: Escalating the Outcome of Attacks against RAG-based Inference in Scale and Severity Using Jailbreaking**

cs.CR

for Github, see  https://github.com/StavC/UnleashingWorms-ExtractingData

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08045v1) [paper-pdf](http://arxiv.org/pdf/2409.08045v1)

**Authors**: Stav Cohen, Ron Bitton, Ben Nassi

**Abstract**: In this paper, we show that with the ability to jailbreak a GenAI model, attackers can escalate the outcome of attacks against RAG-based GenAI-powered applications in severity and scale. In the first part of the paper, we show that attackers can escalate RAG membership inference attacks and RAG entity extraction attacks to RAG documents extraction attacks, forcing a more severe outcome compared to existing attacks. We evaluate the results obtained from three extraction methods, the influence of the type and the size of five embeddings algorithms employed, the size of the provided context, and the GenAI engine. We show that attackers can extract 80%-99.8% of the data stored in the database used by the RAG of a Q&A chatbot. In the second part of the paper, we show that attackers can escalate the scale of RAG data poisoning attacks from compromising a single GenAI-powered application to compromising the entire GenAI ecosystem, forcing a greater scale of damage. This is done by crafting an adversarial self-replicating prompt that triggers a chain reaction of a computer worm within the ecosystem and forces each affected application to perform a malicious activity and compromise the RAG of additional applications. We evaluate the performance of the worm in creating a chain of confidential data extraction about users within a GenAI ecosystem of GenAI-powered email assistants and analyze how the performance of the worm is affected by the size of the context, the adversarial self-replicating prompt used, the type and size of the embeddings algorithm employed, and the number of hops in the propagation. Finally, we review and analyze guardrails to protect RAG-based inference and discuss the tradeoffs.



## **4. Detecting and Defending Against Adversarial Attacks on Automatic Speech Recognition via Diffusion Models**

eess.AS

Under review at ICASSP 2025

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.07936v1) [paper-pdf](http://arxiv.org/pdf/2409.07936v1)

**Authors**: Nikolai L. Kühne, Astrid H. F. Kitchen, Marie S. Jensen, Mikkel S. L. Brøndt, Martin Gonzalez, Christophe Biscio, Zheng-Hua Tan

**Abstract**: Automatic speech recognition (ASR) systems are known to be vulnerable to adversarial attacks. This paper addresses detection and defence against targeted white-box attacks on speech signals for ASR systems. While existing work has utilised diffusion models (DMs) to purify adversarial examples, achieving state-of-the-art results in keyword spotting tasks, their effectiveness for more complex tasks such as sentence-level ASR remains unexplored. Additionally, the impact of the number of forward diffusion steps on performance is not well understood. In this paper, we systematically investigate the use of DMs for defending against adversarial attacks on sentences and examine the effect of varying forward diffusion steps. Through comprehensive experiments on the Mozilla Common Voice dataset, we demonstrate that two forward diffusion steps can completely defend against adversarial attacks on sentences. Moreover, we introduce a novel, training-free approach for detecting adversarial attacks by leveraging a pre-trained DM. Our experimental results show that this method can detect adversarial attacks with high accuracy.



## **5. What Matters to Enhance Traffic Rule Compliance of Imitation Learning for End-to-End Autonomous Driving**

cs.CV

14 pages, 3 figures

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2309.07808v3) [paper-pdf](http://arxiv.org/pdf/2309.07808v3)

**Authors**: Hongkuan Zhou, Wei Cao, Aifen Sui, Zhenshan Bing

**Abstract**: End-to-end autonomous driving, where the entire driving pipeline is replaced with a single neural network, has recently gained research attention because of its simpler structure and faster inference time. Despite this appealing approach largely reducing the complexity in the driving pipeline, it also leads to safety issues because the trained policy is not always compliant with the traffic rules. In this paper, we proposed P-CSG, a penalty-based imitation learning approach with contrastive-based cross semantics generation sensor fusion technologies to increase the overall performance of end-to-end autonomous driving. In this method, we introduce three penalties - red light, stop sign, and curvature speed penalty to make the agent more sensitive to traffic rules. The proposed cross semantics generation helps to align the shared information of different input modalities. We assessed our model's performance using the CARLA Leaderboard - Town 05 Long Benchmark and Longest6 Benchmark, achieving 8.5% and 2.0% driving score improvement compared to the baselines. Furthermore, we conducted robustness evaluations against adversarial attacks like FGSM and Dot attacks, revealing a substantial increase in robustness compared to other baseline models. More detailed information can be found at https://hk-zh.github.io/p-csg-plus.



## **6. A Spatiotemporal Stealthy Backdoor Attack against Cooperative Multi-Agent Deep Reinforcement Learning**

cs.AI

6 pages, IEEE Globecom 2024

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.07775v1) [paper-pdf](http://arxiv.org/pdf/2409.07775v1)

**Authors**: Yinbo Yu, Saihao Yan, Jiajia Liu

**Abstract**: Recent studies have shown that cooperative multi-agent deep reinforcement learning (c-MADRL) is under the threat of backdoor attacks. Once a backdoor trigger is observed, it will perform abnormal actions leading to failures or malicious goals. However, existing proposed backdoors suffer from several issues, e.g., fixed visual trigger patterns lack stealthiness, the backdoor is trained or activated by an additional network, or all agents are backdoored. To this end, in this paper, we propose a novel backdoor attack against c-MADRL, which attacks the entire multi-agent team by embedding the backdoor only in a single agent. Firstly, we introduce adversary spatiotemporal behavior patterns as the backdoor trigger rather than manual-injected fixed visual patterns or instant status and control the attack duration. This method can guarantee the stealthiness and practicality of injected backdoors. Secondly, we hack the original reward function of the backdoored agent via reward reverse and unilateral guidance during training to ensure its adverse influence on the entire team. We evaluate our backdoor attacks on two classic c-MADRL algorithms VDN and QMIX, in a popular c-MADRL environment SMAC. The experimental results demonstrate that our backdoor attacks are able to reach a high attack success rate (91.6\%) while maintaining a low clean performance variance rate (3.7\%).



## **7. Attack End-to-End Autonomous Driving through Module-Wise Noise**

cs.LG

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.07706v1) [paper-pdf](http://arxiv.org/pdf/2409.07706v1)

**Authors**: Lu Wang, Tianyuan Zhang, Yikai Han, Muyang Fang, Ting Jin, Jiaqi Kang

**Abstract**: With recent breakthroughs in deep neural networks, numerous tasks within autonomous driving have exhibited remarkable performance. However, deep learning models are susceptible to adversarial attacks, presenting significant security risks to autonomous driving systems. Presently, end-to-end architectures have emerged as the predominant solution for autonomous driving, owing to their collaborative nature across different tasks. Yet, the implications of adversarial attacks on such models remain relatively unexplored. In this paper, we conduct comprehensive adversarial security research on the modular end-to-end autonomous driving model for the first time. We thoroughly consider the potential vulnerabilities in the model inference process and design a universal attack scheme through module-wise noise injection. We conduct large-scale experiments on the full-stack autonomous driving model and demonstrate that our attack method outperforms previous attack methods. We trust that our research will offer fresh insights into ensuring the safety and reliability of autonomous driving systems.



## **8. A Training Rate and Survival Heuristic for Inference and Robustness Evaluation (TRASHFIRE)**

cs.LG

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2401.13751v2) [paper-pdf](http://arxiv.org/pdf/2401.13751v2)

**Authors**: Charles Meyers, Mohammad Reza Saleh Sedghpour, Tommy Löfstedt, Erik Elmroth

**Abstract**: Machine learning models -- deep neural networks in particular -- have performed remarkably well on benchmark datasets across a wide variety of domains. However, the ease of finding adversarial counter-examples remains a persistent problem when training times are measured in hours or days and the time needed to find a successful adversarial counter-example is measured in seconds. Much work has gone into generating and defending against these adversarial counter-examples, however the relative costs of attacks and defences are rarely discussed. Additionally, machine learning research is almost entirely guided by test/train metrics, but these would require billions of samples to meet industry standards. The present work addresses the problem of understanding and predicting how particular model hyper-parameters influence the performance of a model in the presence of an adversary. The proposed approach uses survival models, worst-case examples, and a cost-aware analysis to precisely and accurately reject a particular model change during routine model training procedures rather than relying on real-world deployment, expensive formal verification methods, or accurate simulations of very complicated systems (\textit{e.g.}, digitally recreating every part of a car or a plane). Through an evaluation of many pre-processing techniques, adversarial counter-examples, and neural network configurations, the conclusion is that deeper models do offer marginal gains in survival times compared to more shallow counterparts. However, we show that those gains are driven more by the model inference time than inherent robustness properties. Using the proposed methodology, we show that ResNet is hopelessly insecure against even the simplest of white box attacks.



## **9. A Cost-Aware Approach to Adversarial Robustness in Neural Networks**

cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07609v1) [paper-pdf](http://arxiv.org/pdf/2409.07609v1)

**Authors**: Charles Meyers, Mohammad Reza Saleh Sedghpour, Tommy Löfstedt, Erik Elmroth

**Abstract**: Considering the growing prominence of production-level AI and the threat of adversarial attacks that can evade a model at run-time, evaluating the robustness of models to these evasion attacks is of critical importance. Additionally, testing model changes likely means deploying the models to (e.g. a car or a medical imaging device), or a drone to see how it affects performance, making un-tested changes a public problem that reduces development speed, increases cost of development, and makes it difficult (if not impossible) to parse cause from effect. In this work, we used survival analysis as a cloud-native, time-efficient and precise method for predicting model performance in the presence of adversarial noise. For neural networks in particular, the relationships between the learning rate, batch size, training time, convergence time, and deployment cost are highly complex, so researchers generally rely on benchmark datasets to assess the ability of a model to generalize beyond the training data. To address this, we propose using accelerated failure time models to measure the effect of hardware choice, batch size, number of epochs, and test-set accuracy by using adversarial attacks to induce failures on a reference model architecture before deploying the model to the real world. We evaluate several GPU types and use the Tree Parzen Estimator to maximize model robustness and minimize model run-time simultaneously. This provides a way to evaluate the model and optimise it in a single step, while simultaneously allowing us to model the effect of model parameters on training time, prediction time, and accuracy. Using this technique, we demonstrate that newer, more-powerful hardware does decrease the training time, but with a monetary and power cost that far outpaces the marginal gains in accuracy.



## **10. Resilient Graph Neural Networks: A Coupled Dynamical Systems Approach**

cs.LG

ECAI 2024

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2311.06942v3) [paper-pdf](http://arxiv.org/pdf/2311.06942v3)

**Authors**: Moshe Eliasof, Davide Murari, Ferdia Sherry, Carola-Bibiane Schönlieb

**Abstract**: Graph Neural Networks (GNNs) have established themselves as a key component in addressing diverse graph-based tasks. Despite their notable successes, GNNs remain susceptible to input perturbations in the form of adversarial attacks. This paper introduces an innovative approach to fortify GNNs against adversarial perturbations through the lens of coupled dynamical systems. Our method introduces graph neural layers based on differential equations with contractive properties, which, as we show, improve the robustness of GNNs. A distinctive feature of the proposed approach is the simultaneous learned evolution of both the node features and the adjacency matrix, yielding an intrinsic enhancement of model robustness to perturbations in the input features and the connectivity of the graph. We mathematically derive the underpinnings of our novel architecture and provide theoretical insights to reason about its expected behavior. We demonstrate the efficacy of our method through numerous real-world benchmarks, reading on par or improved performance compared to existing methods.



## **11. Introducing Perturb-ability Score (PS) to Enhance Robustness Against Evasion Adversarial Attacks on ML-NIDS**

cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07448v1) [paper-pdf](http://arxiv.org/pdf/2409.07448v1)

**Authors**: Mohamed elShehaby, Ashraf Matrawy

**Abstract**: This paper proposes a novel Perturb-ability Score (PS) that can be used to identify Network Intrusion Detection Systems (NIDS) features that can be easily manipulated by attackers in the problem-space. We demonstrate that using PS to select only non-perturb-able features for ML-based NIDS maintains detection performance while enhancing robustness against adversarial attacks.



## **12. Enhancing adversarial robustness in Natural Language Inference using explanations**

cs.CL

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07423v1) [paper-pdf](http://arxiv.org/pdf/2409.07423v1)

**Authors**: Alexandros Koulakos, Maria Lymperaiou, Giorgos Filandrianos, Giorgos Stamou

**Abstract**: The surge of state-of-the-art Transformer-based models has undoubtedly pushed the limits of NLP model performance, excelling in a variety of tasks. We cast the spotlight on the underexplored task of Natural Language Inference (NLI), since models trained on popular well-suited datasets are susceptible to adversarial attacks, allowing subtle input interventions to mislead the model. In this work, we validate the usage of natural language explanation as a model-agnostic defence strategy through extensive experimentation: only by fine-tuning a classifier on the explanation rather than premise-hypothesis inputs, robustness under various adversarial attacks is achieved in comparison to explanation-free baselines. Moreover, since there is no standard strategy of testing the semantic validity of the generated explanations, we research the correlation of widely used language generation metrics with human perception, in order for them to serve as a proxy towards robust NLI models. Our approach is resource-efficient and reproducible without significant computational limitations.



## **13. SoK: Security and Privacy Risks of Medical AI**

cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07415v1) [paper-pdf](http://arxiv.org/pdf/2409.07415v1)

**Authors**: Yuanhaur Chang, Han Liu, Evin Jaff, Chenyang Lu, Ning Zhang

**Abstract**: The integration of technology and healthcare has ushered in a new era where software systems, powered by artificial intelligence and machine learning, have become essential components of medical products and services. While these advancements hold great promise for enhancing patient care and healthcare delivery efficiency, they also expose sensitive medical data and system integrity to potential cyberattacks. This paper explores the security and privacy threats posed by AI/ML applications in healthcare. Through a thorough examination of existing research across a range of medical domains, we have identified significant gaps in understanding the adversarial attacks targeting medical AI systems. By outlining specific adversarial threat models for medical settings and identifying vulnerable application domains, we lay the groundwork for future research that investigates the security and resilience of AI-driven medical systems. Through our analysis of different threat models and feasibility studies on adversarial attacks in different medical domains, we provide compelling insights into the pressing need for cybersecurity research in the rapidly evolving field of AI healthcare technology.



## **14. D-CAPTCHA++: A Study of Resilience of Deepfake CAPTCHA under Transferable Imperceptible Adversarial Attack**

cs.CR

14 pages

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07390v1) [paper-pdf](http://arxiv.org/pdf/2409.07390v1)

**Authors**: Hong-Hanh Nguyen-Le, Van-Tuan Tran, Dinh-Thuc Nguyen, Nhien-An Le-Khac

**Abstract**: The advancements in generative AI have enabled the improvement of audio synthesis models, including text-to-speech and voice conversion. This raises concerns about its potential misuse in social manipulation and political interference, as synthetic speech has become indistinguishable from natural human speech. Several speech-generation programs are utilized for malicious purposes, especially impersonating individuals through phone calls. Therefore, detecting fake audio is crucial to maintain social security and safeguard the integrity of information. Recent research has proposed a D-CAPTCHA system based on the challenge-response protocol to differentiate fake phone calls from real ones. In this work, we study the resilience of this system and introduce a more robust version, D-CAPTCHA++, to defend against fake calls. Specifically, we first expose the vulnerability of the D-CAPTCHA system under transferable imperceptible adversarial attack. Secondly, we mitigate such vulnerability by improving the robustness of the system by using adversarial training in D-CAPTCHA deepfake detectors and task classifiers.



## **15. Securing Vision-Language Models with a Robust Encoder Against Jailbreak and Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07353v1) [paper-pdf](http://arxiv.org/pdf/2409.07353v1)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Large Vision-Language Models (LVLMs), trained on multimodal big datasets, have significantly advanced AI by excelling in vision-language tasks. However, these models remain vulnerable to adversarial attacks, particularly jailbreak attacks, which bypass safety protocols and cause the model to generate misleading or harmful responses. This vulnerability stems from both the inherent susceptibilities of LLMs and the expanded attack surface introduced by the visual modality. We propose Sim-CLIP+, a novel defense mechanism that adversarially fine-tunes the CLIP vision encoder by leveraging a Siamese architecture. This approach maximizes cosine similarity between perturbed and clean samples, facilitating resilience against adversarial manipulations. Sim-CLIP+ offers a plug-and-play solution, allowing seamless integration into existing LVLM architectures as a robust vision encoder. Unlike previous defenses, our method requires no structural modifications to the LVLM and incurs minimal computational overhead. Sim-CLIP+ demonstrates effectiveness against both gradient-based adversarial attacks and various jailbreak techniques. We evaluate Sim-CLIP+ against three distinct jailbreak attack strategies and perform clean evaluations using standard downstream datasets, including COCO for image captioning and OKVQA for visual question answering. Extensive experiments demonstrate that Sim-CLIP+ maintains high clean accuracy while substantially improving robustness against both gradient-based adversarial attacks and jailbreak techniques. Our code and robust vision encoders are available at https://github.com/speedlab-git/Robust-Encoder-against-Jailbreak-attack.git.



## **16. Module-wise Adaptive Adversarial Training for End-to-end Autonomous Driving**

cs.CV

14 pages

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07321v1) [paper-pdf](http://arxiv.org/pdf/2409.07321v1)

**Authors**: Tianyuan Zhang, Lu Wang, Jiaqi Kang, Xinwei Zhang, Siyuan Liang, Yuwei Chen, Aishan Liu, Xianglong Liu

**Abstract**: Recent advances in deep learning have markedly improved autonomous driving (AD) models, particularly end-to-end systems that integrate perception, prediction, and planning stages, achieving state-of-the-art performance. However, these models remain vulnerable to adversarial attacks, where human-imperceptible perturbations can disrupt decision-making processes. While adversarial training is an effective method for enhancing model robustness against such attacks, no prior studies have focused on its application to end-to-end AD models. In this paper, we take the first step in adversarial training for end-to-end AD models and present a novel Module-wise Adaptive Adversarial Training (MA2T). However, extending conventional adversarial training to this context is highly non-trivial, as different stages within the model have distinct objectives and are strongly interconnected. To address these challenges, MA2T first introduces Module-wise Noise Injection, which injects noise before the input of different modules, targeting training models with the guidance of overall objectives rather than each independent module loss. Additionally, we introduce Dynamic Weight Accumulation Adaptation, which incorporates accumulated weight changes to adaptively learn and adjust the loss weights of each module based on their contributions (accumulated reduction rates) for better balance and robust training. To demonstrate the efficacy of our defense, we conduct extensive experiments on the widely-used nuScenes dataset across several end-to-end AD models under both white-box and black-box attacks, where our method outperforms other baselines by large margins (+5-10%). Moreover, we validate the robustness of our defense through closed-loop evaluation in the CARLA simulation environment, showing improved resilience even against natural corruption.



## **17. Optimizing Neural Network Performance and Interpretability with Diophantine Equation Encoding**

cs.LG

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07310v1) [paper-pdf](http://arxiv.org/pdf/2409.07310v1)

**Authors**: Ronald Katende

**Abstract**: This paper explores the integration of Diophantine equations into neural network (NN) architectures to improve model interpretability, stability, and efficiency. By encoding and decoding neural network parameters as integer solutions to Diophantine equations, we introduce a novel approach that enhances both the precision and robustness of deep learning models. Our method integrates a custom loss function that enforces Diophantine constraints during training, leading to better generalization, reduced error bounds, and enhanced resilience against adversarial attacks. We demonstrate the efficacy of this approach through several tasks, including image classification and natural language processing, where improvements in accuracy, convergence, and robustness are observed. This study offers a new perspective on combining mathematical theory and machine learning to create more interpretable and efficient models.



## **18. Potion: Towards Poison Unlearning**

cs.LG

Accepted for publication in the Journal of Data-centric Machine  Learning Research (DMLR) https://openreview.net/forum?id=4eSiRnWWaF

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2406.09173v3) [paper-pdf](http://arxiv.org/pdf/2406.09173v3)

**Authors**: Stefan Schoepf, Jack Foster, Alexandra Brintrup

**Abstract**: Adversarial attacks by malicious actors on machine learning systems, such as introducing poison triggers into training datasets, pose significant risks. The challenge in resolving such an attack arises in practice when only a subset of the poisoned data can be identified. This necessitates the development of methods to remove, i.e. unlearn, poison triggers from already trained models with only a subset of the poison data available. The requirements for this task significantly deviate from privacy-focused unlearning where all of the data to be forgotten by the model is known. Previous work has shown that the undiscovered poisoned samples lead to a failure of established unlearning methods, with only one method, Selective Synaptic Dampening (SSD), showing limited success. Even full retraining, after the removal of the identified poison, cannot address this challenge as the undiscovered poison samples lead to a reintroduction of the poison trigger in the model. Our work addresses two key challenges to advance the state of the art in poison unlearning. First, we introduce a novel outlier-resistant method, based on SSD, that significantly improves model protection and unlearning performance. Second, we introduce Poison Trigger Neutralisation (PTN) search, a fast, parallelisable, hyperparameter search that utilises the characteristic "unlearning versus model protection" trade-off to find suitable hyperparameters in settings where the forget set size is unknown and the retain set is contaminated. We benchmark our contributions using ResNet-9 on CIFAR10 and WideResNet-28x10 on CIFAR100. Experimental results show that our method heals 93.72% of poison compared to SSD with 83.41% and full retraining with 40.68%. We achieve this while also lowering the average model accuracy drop caused by unlearning from 5.68% (SSD) to 1.41% (ours).



## **19. Countering adversarial perturbations in graphs using error correcting codes**

cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2406.14245v2) [paper-pdf](http://arxiv.org/pdf/2406.14245v2)

**Authors**: Saif Eddin Jabari

**Abstract**: We consider the problem of a graph subjected to adversarial perturbations, such as those arising from cyber-attacks, where edges are covertly added or removed. The adversarial perturbations occur during the transmission of the graph between a sender and a receiver. To counteract potential perturbations, this study explores a repetition coding scheme with sender-assigned noise and majority voting on the receiver's end to rectify the graph's structure. The approach operates without prior knowledge of the attack's characteristics. We analytically derive a bound on the number of repetitions needed to satisfy probabilistic constraints on the quality of the reconstructed graph. The method can accurately and effectively decode Erd\H{o}s-R\'{e}nyi graphs that were subjected to non-random edge removal, namely, those connected to vertices with the highest eigenvector centrality, in addition to random addition and removal of edges by the attacker. The method is also effective against attacks on large scale-free graphs generated using the Barab\'{a}si-Albert model but require a larger number of repetitions than needed to correct Erd\H{o}s-R\'{e}nyi graphs.



## **20. The Philosopher's Stone: Trojaning Plugins of Large Language Models**

cs.CR

Accepted by NDSS Symposium 2025. Please cite this paper as "Tian  Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Yan Meng, Shaofeng Li, Zhen  Liu, Haojin Zhu. The Philosopher's Stone: Trojaning Plugins of Large Language  Models. In the 32nd Annual Network and Distributed System Security Symposium  (NDSS 2025)."

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2312.00374v3) [paper-pdf](http://arxiv.org/pdf/2312.00374v3)

**Authors**: Tian Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Yan Meng, Shaofeng Li, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers,an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses a superior LLM to align na\"ively poisoned data based on our insight that it can better inject poisoning knowledge during training. In contrast, FUSION leverages a novel over-poisoning procedure to transform a benign adapter into a malicious one by magnifying the attention between trigger and target in model weights. In our experiments, we first conduct two case studies to demonstrate that a compromised LLM agent can use malware to control the system (e.g., a LLM-driven robot) or to launch a spear-phishing attack. Then, in terms of targeted misinformation, we show that our attacks provide higher attack effectiveness than the existing baseline and, for the purpose of attracting downloads, preserve or improve the adapter's utility. Finally, we designed and evaluated three potential defenses. However, none proved entirely effective in safeguarding against our attacks, highlighting the need for more robust defenses supporting a secure LLM supply chain.



## **21. FullCert: Deterministic End-to-End Certification for Training and Inference of Neural Networks**

cs.LG

This preprint has not undergone peer review or any post-submission  improvements or corrections. The Version of Record of this contribution is  published in DAGM GCPR 2024

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2406.11522v2) [paper-pdf](http://arxiv.org/pdf/2406.11522v2)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstract**: Modern machine learning models are sensitive to the manipulation of both the training data (poisoning attacks) and inference data (adversarial examples). Recognizing this issue, the community has developed many empirical defenses against both attacks and, more recently, certification methods with provable guarantees against inference-time attacks. However, such guarantees are still largely lacking for training-time attacks. In this work, we present FullCert, the first end-to-end certifier with sound, deterministic bounds, which proves robustness against both training-time and inference-time attacks. We first bound all possible perturbations an adversary can make to the training data under the considered threat model. Using these constraints, we bound the perturbations' influence on the model's parameters. Finally, we bound the impact of these parameter changes on the model's prediction, resulting in joint robustness guarantees against poisoning and adversarial examples. To facilitate this novel certification paradigm, we combine our theoretical work with a new open-source library BoundFlow, which enables model training on bounded datasets. We experimentally demonstrate FullCert's feasibility on two datasets.



## **22. Adversarial Doodles: Interpretable and Human-drawable Attacks Provide Describable Insights**

cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2311.15994v3) [paper-pdf](http://arxiv.org/pdf/2311.15994v3)

**Authors**: Ryoya Nara, Yusuke Matsui

**Abstract**: DNN-based image classifiers are susceptible to adversarial attacks. Most previous adversarial attacks do not have clear patterns, making it difficult to interpret attacks' results and gain insights into classifiers' mechanisms. Therefore, we propose Adversarial Doodles, which have interpretable shapes. We optimize black bezier curves to fool the classifier by overlaying them onto the input image. By introducing random affine transformation and regularizing the doodled area, we obtain small-sized attacks that cause misclassification even when humans replicate them by hand. Adversarial doodles provide describable insights into the relationship between the human-drawn doodle's shape and the classifier's output, such as "When we add three small circles on a helicopter image, the ResNet-50 classifier mistakenly classifies it as an airplane."



## **23. Privacy-oriented manipulation of speaker representations**

eess.AS

Article published in IEEE Access

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2310.06652v2) [paper-pdf](http://arxiv.org/pdf/2310.06652v2)

**Authors**: Francisco Teixeira, Alberto Abad, Bhiksha Raj, Isabel Trancoso

**Abstract**: Speaker embeddings are ubiquitous, with applications ranging from speaker recognition and diarization to speech synthesis and voice anonymisation. The amount of information held by these embeddings lends them versatility, but also raises privacy concerns. Speaker embeddings have been shown to contain information on age, sex, health and more, which speakers may want to keep private, especially when this information is not required for the target task. In this work, we propose a method for removing and manipulating private attributes from speaker embeddings that leverages a Vector-Quantized Variational Autoencoder architecture, combined with an adversarial classifier and a novel mutual information loss. We validate our model on two attributes, sex and age, and perform experiments with ignorant and fully-informed attackers, and with in-domain and out-of-domain data.



## **24. Attack on Scene Flow using Point Clouds**

cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2404.13621v6) [paper-pdf](http://arxiv.org/pdf/2404.13621v6)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. The robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants shows a higher vulnerability for the optical flow networks. Code is available at https://github.com/aheldis/Attack-on-Scene-Flow-using-Point-Clouds.git.



## **25. CPSample: Classifier Protected Sampling for Guarding Training Data During Diffusion**

cs.LG

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07025v1) [paper-pdf](http://arxiv.org/pdf/2409.07025v1)

**Authors**: Joshua Kazdan, Hao Sun, Jiaqi Han, Felix Petersen, Stefano Ermon

**Abstract**: Diffusion models have a tendency to exactly replicate their training data, especially when trained on small datasets. Most prior work has sought to mitigate this problem by imposing differential privacy constraints or masking parts of the training data, resulting in a notable substantial decrease in image quality. We present CPSample, a method that modifies the sampling process to prevent training data replication while preserving image quality. CPSample utilizes a classifier that is trained to overfit on random binary labels attached to the training data. CPSample then uses classifier guidance to steer the generation process away from the set of points that can be classified with high certainty, a set that includes the training data. CPSample achieves FID scores of 4.97 and 2.97 on CIFAR-10 and CelebA-64, respectively, without producing exact replicates of the training data. Unlike prior methods intended to guard the training images, CPSample only requires training a classifier rather than retraining a diffusion model, which is computationally cheaper. Moreover, our technique provides diffusion models with greater robustness against membership inference attacks, wherein an adversary attempts to discern which images were in the model's training dataset. We show that CPSample behaves like a built-in rejection sampler, and we demonstrate its capabilities to prevent mode collapse in Stable Diffusion.



## **26. AdvLogo: Adversarial Patch Attack against Object Detectors based on Diffusion Models**

cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07002v1) [paper-pdf](http://arxiv.org/pdf/2409.07002v1)

**Authors**: Boming Miao, Chunxiao Li, Yao Zhu, Weixiang Sun, Zizhe Wang, Xiaoyi Wang, Chuanlong Xie

**Abstract**: With the rapid development of deep learning, object detectors have demonstrated impressive performance; however, vulnerabilities still exist in certain scenarios. Current research exploring the vulnerabilities using adversarial patches often struggles to balance the trade-off between attack effectiveness and visual quality. To address this problem, we propose a novel framework of patch attack from semantic perspective, which we refer to as AdvLogo. Based on the hypothesis that every semantic space contains an adversarial subspace where images can cause detectors to fail in recognizing objects, we leverage the semantic understanding of the diffusion denoising process and drive the process to adversarial subareas by perturbing the latent and unconditional embeddings at the last timestep. To mitigate the distribution shift that exposes a negative impact on image quality, we apply perturbation to the latent in frequency domain with the Fourier Transform. Experimental results demonstrate that AdvLogo achieves strong attack performance while maintaining high visual quality.



## **27. Privacy Leakage on DNNs: A Survey of Model Inversion Attacks and Defenses**

cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2402.04013v2) [paper-pdf](http://arxiv.org/pdf/2402.04013v2)

**Authors**: Hao Fang, Yixiang Qiu, Hongyao Yu, Wenbo Yu, Jiawei Kong, Baoli Chong, Bin Chen, Xuan Wang, Shu-Tao Xia, Ke Xu

**Abstract**: Deep Neural Networks (DNNs) have revolutionized various domains with their exceptional performance across numerous applications. However, Model Inversion (MI) attacks, which disclose private information about the training dataset by abusing access to the trained models, have emerged as a formidable privacy threat. Given a trained network, these attacks enable adversaries to reconstruct high-fidelity data that closely aligns with the private training samples, posing significant privacy concerns. Despite the rapid advances in the field, we lack a comprehensive and systematic overview of existing MI attacks and defenses. To fill this gap, this paper thoroughly investigates this realm and presents a holistic survey. Firstly, our work briefly reviews early MI studies on traditional machine learning scenarios. We then elaborately analyze and compare numerous recent attacks and defenses on Deep Neural Networks (DNNs) across multiple modalities and learning tasks. By meticulously analyzing their distinctive features, we summarize and classify these methods into different categories and provide a novel taxonomy. Finally, this paper discusses promising research directions and presents potential solutions to open issues. To facilitate further study on MI attacks and defenses, we have implemented an open-source model inversion toolbox on GitHub (https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox).



## **28. Well, that escalated quickly: The Single-Turn Crescendo Attack (STCA)**

cs.CR

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.03131v2) [paper-pdf](http://arxiv.org/pdf/2409.03131v2)

**Authors**: Alan Aqrawi, Arian Abbasi

**Abstract**: This paper introduces a new method for adversarial attacks on large language models (LLMs) called the Single-Turn Crescendo Attack (STCA). Building on the multi-turn crescendo attack method introduced by Russinovich, Salem, and Eldan (2024), which gradually escalates the context to provoke harmful responses, the STCA achieves similar outcomes in a single interaction. By condensing the escalation into a single, well-crafted prompt, the STCA bypasses typical moderation filters that LLMs use to prevent inappropriate outputs. This technique reveals vulnerabilities in current LLMs and emphasizes the importance of stronger safeguards in responsible AI (RAI). The STCA offers a novel method that has not been previously explored.



## **29. Personalized Federated Learning Techniques: Empirical Analysis**

cs.LG

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.06805v1) [paper-pdf](http://arxiv.org/pdf/2409.06805v1)

**Authors**: Azal Ahmad Khan, Ahmad Faraz Khan, Haider Ali, Ali Anwar

**Abstract**: Personalized Federated Learning (pFL) holds immense promise for tailoring machine learning models to individual users while preserving data privacy. However, achieving optimal performance in pFL often requires a careful balancing act between memory overhead costs and model accuracy. This paper delves into the trade-offs inherent in pFL, offering valuable insights for selecting the right algorithms for diverse real-world scenarios. We empirically evaluate ten prominent pFL techniques across various datasets and data splits, uncovering significant differences in their performance. Our study reveals interesting insights into how pFL methods that utilize personalized (local) aggregation exhibit the fastest convergence due to their efficiency in communication and computation. Conversely, fine-tuning methods face limitations in handling data heterogeneity and potential adversarial attacks while multi-objective learning methods achieve higher accuracy at the cost of additional training and resource consumption. Our study emphasizes the critical role of communication efficiency in scaling pFL, demonstrating how it can significantly affect resource usage in real-world deployments.



## **30. Adversarial Attacks to Multi-Modal Models**

cs.CR

To appear in the ACM Workshop on Large AI Systems and Models with  Privacy and Safety Analysis 2024 (LAMPS '24)

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.06793v1) [paper-pdf](http://arxiv.org/pdf/2409.06793v1)

**Authors**: Zhihao Dou, Xin Hu, Haibo Yang, Zhuqing Liu, Minghong Fang

**Abstract**: Multi-modal models have gained significant attention due to their powerful capabilities. These models effectively align embeddings across diverse data modalities, showcasing superior performance in downstream tasks compared to their unimodal counterparts. Recent study showed that the attacker can manipulate an image or audio file by altering it in such a way that its embedding matches that of an attacker-chosen targeted input, thereby deceiving downstream models. However, this method often underperforms due to inherent disparities in data from different modalities. In this paper, we introduce CrossFire, an innovative approach to attack multi-modal models. CrossFire begins by transforming the targeted input chosen by the attacker into a format that matches the modality of the original image or audio file. We then formulate our attack as an optimization problem, aiming to minimize the angular deviation between the embeddings of the transformed input and the modified image or audio file. Solving this problem determines the perturbations to be added to the original media. Our extensive experiments on six real-world benchmark datasets reveal that CrossFire can significantly manipulate downstream tasks, surpassing existing attacks. Additionally, we evaluate six defensive strategies against CrossFire, finding that current defenses are insufficient to counteract our CrossFire.



## **31. DNN-Defender: A Victim-Focused In-DRAM Defense Mechanism for Taming Adversarial Weight Attack on DNNs**

cs.CR

6 pages, 9 figures

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2305.08034v2) [paper-pdf](http://arxiv.org/pdf/2305.08034v2)

**Authors**: Ranyang Zhou, Sabbir Ahmed, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: With deep learning deployed in many security-sensitive areas, machine learning security is becoming progressively important. Recent studies demonstrate attackers can exploit system-level techniques exploiting the RowHammer vulnerability of DRAM to deterministically and precisely flip bits in Deep Neural Networks (DNN) model weights to affect inference accuracy. The existing defense mechanisms are software-based, such as weight reconstruction requiring expensive training overhead or performance degradation. On the other hand, generic hardware-based victim-/aggressor-focused mechanisms impose expensive hardware overheads and preserve the spatial connection between victim and aggressor rows. In this paper, we present the first DRAM-based victim-focused defense mechanism tailored for quantized DNNs, named DNN-Defender that leverages the potential of in-DRAM swapping to withstand the targeted bit-flip attacks with a priority protection mechanism. Our results indicate that DNN-Defender can deliver a high level of protection downgrading the performance of targeted RowHammer attacks to a random attack level. In addition, the proposed defense has no accuracy drop on CIFAR-10 and ImageNet datasets without requiring any software training or incurring hardware overhead.



## **32. DV-FSR: A Dual-View Target Attack Framework for Federated Sequential Recommendation**

cs.CR

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.07500v1) [paper-pdf](http://arxiv.org/pdf/2409.07500v1)

**Authors**: Qitao Qin, Yucong Luo, Mingyue Cheng, Qingyang Mao, Chenyi Lei

**Abstract**: Federated recommendation (FedRec) preserves user privacy by enabling decentralized training of personalized models, but this architecture is inherently vulnerable to adversarial attacks. Significant research has been conducted on targeted attacks in FedRec systems, motivated by commercial and social influence considerations. However, much of this work has largely overlooked the differential robustness of recommendation models. Moreover, our empirical findings indicate that existing targeted attack methods achieve only limited effectiveness in Federated Sequential Recommendation (FSR) tasks. Driven by these observations, we focus on investigating targeted attacks in FSR and propose a novel dualview attack framework, named DV-FSR. This attack method uniquely combines a sampling-based explicit strategy with a contrastive learning-based implicit gradient strategy to orchestrate a coordinated attack. Additionally, we introduce a specific defense mechanism tailored for targeted attacks in FSR, aiming to evaluate the mitigation effects of the attack method we proposed. Extensive experiments validate the effectiveness of our proposed approach on representative sequential models.



## **33. Shedding More Light on Robust Classifiers under the lens of Energy-based Models**

cs.CV

Accepted at European Conference on Computer Vision (ECCV) 2024

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2407.06315v3) [paper-pdf](http://arxiv.org/pdf/2407.06315v3)

**Authors**: Mujtaba Hussain Mirza, Maria Rosaria Briglia, Senad Beadini, Iacopo Masi

**Abstract**: By reinterpreting a robust discriminative classifier as Energy-based Model (EBM), we offer a new take on the dynamics of adversarial training (AT). Our analysis of the energy landscape during AT reveals that untargeted attacks generate adversarial images much more in-distribution (lower energy) than the original data from the point of view of the model. Conversely, we observe the opposite for targeted attacks. On the ground of our thorough analysis, we present new theoretical and practical results that show how interpreting AT energy dynamics unlocks a better understanding: (1) AT dynamic is governed by three phases and robust overfitting occurs in the third phase with a drastic divergence between natural and adversarial energies (2) by rewriting the loss of TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization (TRADES) in terms of energies, we show that TRADES implicitly alleviates overfitting by means of aligning the natural energy with the adversarial one (3) we empirically show that all recent state-of-the-art robust classifiers are smoothing the energy landscape and we reconcile a variety of studies about understanding AT and weighting the loss function under the umbrella of EBMs. Motivated by rigorous evidence, we propose Weighted Energy Adversarial Training (WEAT), a novel sample weighting scheme that yields robust accuracy matching the state-of-the-art on multiple benchmarks such as CIFAR-10 and SVHN and going beyond in CIFAR-100 and Tiny-ImageNet. We further show that robust classifiers vary in the intensity and quality of their generative capabilities, and offer a simple method to push this capability, reaching a remarkable Inception Score (IS) and FID using a robust classifier without training for generative modeling. The code to reproduce our results is available at http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/ .



## **34. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

cs.DC

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2404.14250v5) [paper-pdf](http://arxiv.org/pdf/2404.14250v5)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.



## **35. Unrevealed Threats: A Comprehensive Study of the Adversarial Robustness of Underwater Image Enhancement Models**

eess.IV

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.06420v1) [paper-pdf](http://arxiv.org/pdf/2409.06420v1)

**Authors**: Siyu Zhai, Zhibo He, Xiaofeng Cong, Junming Hou, Jie Gui, Jian Wei You, Xin Gong, James Tin-Yau Kwok, Yuan Yan Tang

**Abstract**: Learning-based methods for underwater image enhancement (UWIE) have undergone extensive exploration. However, learning-based models are usually vulnerable to adversarial examples so as the UWIE models. To the best of our knowledge, there is no comprehensive study on the adversarial robustness of UWIE models, which indicates that UWIE models are potentially under the threat of adversarial attacks. In this paper, we propose a general adversarial attack protocol. We make a first attempt to conduct adversarial attacks on five well-designed UWIE models on three common underwater image benchmark datasets. Considering the scattering and absorption of light in the underwater environment, there exists a strong correlation between color correction and underwater image enhancement. On the basis of that, we also design two effective UWIE-oriented adversarial attack methods Pixel Attack and Color Shift Attack targeting different color spaces. The results show that five models exhibit varying degrees of vulnerability to adversarial attacks and well-designed small perturbations on degraded images are capable of preventing UWIE models from generating enhanced results. Further, we conduct adversarial training on these models and successfully mitigated the effectiveness of adversarial attacks. In summary, we reveal the adversarial vulnerability of UWIE models and propose a new evaluation dimension of UWIE models.



## **36. Passive Inference Attacks on Split Learning via Adversarial Regularization**

cs.CR

To appear at NDSS 2025; 25 pages, 27 figures

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2310.10483v5) [paper-pdf](http://arxiv.org/pdf/2310.10483v5)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more capable attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves significantly superior attack performance, even comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.



## **37. Influence-based Attributions can be Manipulated**

cs.LG

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.05208v2) [paper-pdf](http://arxiv.org/pdf/2409.05208v2)

**Authors**: Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri

**Abstract**: Influence Functions are a standard tool for attributing predictions to training data in a principled manner and are widely used in applications such as data valuation and fairness. In this work, we present realistic incentives to manipulate influencebased attributions and investigate whether these attributions can be systematically tampered by an adversary. We show that this is indeed possible and provide efficient attacks with backward-friendly implementations. Our work raises questions on the reliability of influence-based attributions under adversarial circumstances.



## **38. On the Weaknesses of Backdoor-based Model Watermarking: An Information-theoretic Perspective**

cs.CR

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.06130v1) [paper-pdf](http://arxiv.org/pdf/2409.06130v1)

**Authors**: Aoting Hu, Yanzhi Chen, Renjie Xie, Adrian Weller

**Abstract**: Safeguarding the intellectual property of machine learning models has emerged as a pressing concern in AI security. Model watermarking is a powerful technique for protecting ownership of machine learning models, yet its reliability has been recently challenged by recent watermark removal attacks. In this work, we investigate why existing watermark embedding techniques particularly those based on backdooring are vulnerable. Through an information-theoretic analysis, we show that the resilience of watermarking against erasure attacks hinges on the choice of trigger-set samples, where current uses of out-distribution trigger-set are inherently vulnerable to white-box adversaries. Based on this discovery, we propose a novel model watermarking scheme, In-distribution Watermark Embedding (IWE), to overcome the limitations of existing method. To further minimise the gap to clean models, we analyze the role of logits as watermark information carriers and propose a new approach to better conceal watermark information within the logits. Experiments on real-world datasets including CIFAR-100 and Caltech-101 demonstrate that our method robustly defends against various adversaries with negligible accuracy loss (< 0.1%).



## **39. Concealing Backdoor Model Updates in Federated Learning by Trigger-Optimized Data Poisoning**

cs.CR

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2405.06206v2) [paper-pdf](http://arxiv.org/pdf/2405.06206v2)

**Authors**: Yujie Zhang, Neil Gong, Michael K. Reiter

**Abstract**: Federated Learning (FL) is a decentralized machine learning method that enables participants to collaboratively train a model without sharing their private data. Despite its privacy and scalability benefits, FL is susceptible to backdoor attacks, where adversaries poison the local training data of a subset of clients using a backdoor trigger, aiming to make the aggregated model produce malicious results when the same backdoor condition is met by an inference-time input. Existing backdoor attacks in FL suffer from common deficiencies: fixed trigger patterns and reliance on the assistance of model poisoning. State-of-the-art defenses based on analyzing clients' model updates exhibit a good defense performance on these attacks because of the significant divergence between malicious and benign client model updates. To effectively conceal malicious model updates among benign ones, we propose DPOT, a backdoor attack strategy in FL that dynamically constructs backdoor objectives by optimizing a backdoor trigger, making backdoor data have minimal effect on model updates. We provide theoretical justifications for DPOT's attacking principle and display experimental results showing that DPOT, via only a data-poisoning attack, effectively undermines state-of-the-art defenses and outperforms existing backdoor attack techniques on various datasets.



## **40. Cross-Input Certified Training for Universal Perturbations**

cs.LG

23 pages, 6 figures, ECCV '24

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2405.09176v2) [paper-pdf](http://arxiv.org/pdf/2405.09176v2)

**Authors**: Changming Xu, Gagandeep Singh

**Abstract**: Existing work in trustworthy machine learning primarily focuses on single-input adversarial perturbations. In many real-world attack scenarios, input-agnostic adversarial attacks, e.g. universal adversarial perturbations (UAPs), are much more feasible. Current certified training methods train models robust to single-input perturbations but achieve suboptimal clean and UAP accuracy, thereby limiting their applicability in practical applications. We propose a novel method, CITRUS, for certified training of networks robust against UAP attackers. We show in an extensive evaluation across different datasets, architectures, and perturbation magnitudes that our method outperforms traditional certified training methods on standard accuracy (up to 10.3\%) and achieves SOTA performance on the more practical certified UAP accuracy metric.



## **41. Espresso: Robust Concept Filtering in Text-to-Image Models**

cs.CV

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2404.19227v5) [paper-pdf](http://arxiv.org/pdf/2404.19227v5)

**Authors**: Anudeep Das, Vasisht Duddu, Rui Zhang, N. Asokan

**Abstract**: Diffusion based text-to-image models are trained on large datasets scraped from the Internet, potentially containing unacceptable concepts (e.g., copyright infringing or unsafe). We need concept removal techniques (CRTs) which are effective in preventing the generation of images with unacceptable concepts, utility-preserving on acceptable concepts, and robust against evasion with adversarial prompts. None of the prior CRTs satisfy all these requirements simultaneously. We introduce Espresso, the first robust concept filter based on Contrastive Language-Image Pre-Training (CLIP). We configure CLIP to identify unacceptable concepts in generated images using the distance of their embeddings to the text embeddings of both unacceptable and acceptable concepts. This lets us fine-tune for robustness by separating the text embeddings of unacceptable and acceptable concepts while preserving their pairing with image embeddings for utility. We present a pipeline to evaluate various CRTs, attacks against them, and show that Espresso, is more effective and robust than prior CRTs, while retaining utility.



## **42. Unlearning or Concealment? A Critical Analysis and Evaluation Metrics for Unlearning in Diffusion Models**

cs.LG

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.05668v1) [paper-pdf](http://arxiv.org/pdf/2409.05668v1)

**Authors**: Aakash Sen Sharma, Niladri Sarkar, Vikram Chundawat, Ankur A Mali, Murari Mandal

**Abstract**: Recent research has seen significant interest in methods for concept removal and targeted forgetting in diffusion models. In this paper, we conduct a comprehensive white-box analysis to expose significant vulnerabilities in existing diffusion model unlearning methods. We show that the objective functions used for unlearning in the existing methods lead to decoupling of the targeted concepts (meant to be forgotten) for the corresponding prompts. This is concealment and not actual unlearning, which was the original goal. The ineffectiveness of current methods stems primarily from their narrow focus on reducing generation probabilities for specific prompt sets, neglecting the diverse modalities of intermediate guidance employed during the inference process. The paper presents a rigorous theoretical and empirical examination of four commonly used techniques for unlearning in diffusion models. We introduce two new evaluation metrics: Concept Retrieval Score (CRS) and Concept Confidence Score (CCS). These metrics are based on a successful adversarial attack setup that can recover forgotten concepts from unlearned diffusion models. The CRS measures the similarity between the latent representations of the unlearned and fully trained models after unlearning. It reports the extent of retrieval of the forgotten concepts with increasing amount of guidance. The CCS quantifies the confidence of the model in assigning the target concept to the manipulated data. It reports the probability of the unlearned model's generations to be aligned with the original domain knowledge with increasing amount of guidance. Evaluating existing unlearning methods with our proposed stringent metrics for diffusion models reveals significant shortcomings in their ability to truly unlearn concepts. Source Code: https://respailab.github.io/unlearning-or-concealment



## **43. Adversarial Attacks on Data Attribution**

cs.LG

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.05657v1) [paper-pdf](http://arxiv.org/pdf/2409.05657v1)

**Authors**: Xinhe Wang, Pingbang Hu, Junwei Deng, Jiaqi W. Ma

**Abstract**: Data attribution aims to quantify the contribution of individual training data points to the outputs of an AI model, which has been used to measure the value of training data and compensate data providers. Given the impact on financial decisions and compensation mechanisms, a critical question arises concerning the adversarial robustness of data attribution methods. However, there has been little to no systematic research addressing this issue. In this work, we aim to bridge this gap by detailing a threat model with clear assumptions about the adversary's goal and capabilities, and by proposing principled adversarial attack methods on data attribution. We present two such methods, Shadow Attack and Outlier Attack, both of which generate manipulated datasets to adversarially inflate the compensation. The Shadow Attack leverages knowledge about the data distribution in the AI applications, and derives adversarial perturbations through "shadow training", a technique commonly used in membership inference attacks. In contrast, the Outlier Attack does not assume any knowledge about the data distribution and relies solely on black-box queries to the target model's predictions. It exploits an inductive bias present in many data attribution methods - outlier data points are more likely to be influential - and employs adversarial examples to generate manipulated datasets. Empirically, in image classification and text generation tasks, the Shadow Attack can inflate the data-attribution-based compensation by at least 200%, while the Outlier Attack achieves compensation inflation ranging from 185% to as much as 643%.



## **44. A Framework for Differential Privacy Against Timing Attacks**

cs.CR

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.05623v1) [paper-pdf](http://arxiv.org/pdf/2409.05623v1)

**Authors**: Zachary Ratliff, Salil Vadhan

**Abstract**: The standard definition of differential privacy (DP) ensures that a mechanism's output distribution on adjacent datasets is indistinguishable. However, real-world implementations of DP can, and often do, reveal information through their runtime distributions, making them susceptible to timing attacks. In this work, we establish a general framework for ensuring differential privacy in the presence of timing side channels. We define a new notion of timing privacy, which captures programs that remain differentially private to an adversary that observes the program's runtime in addition to the output. Our framework enables chaining together component programs that are timing-stable followed by a random delay to obtain DP programs that achieve timing privacy. Importantly, our definitions allow for measuring timing privacy and output privacy using different privacy measures. We illustrate how to instantiate our framework by giving programs for standard DP computations in the RAM and Word RAM models of computation. Furthermore, we show how our framework can be realized in code through a natural extension of the OpenDP Programming Framework.



## **45. How adversarial attacks can disrupt seemingly stable accurate classifiers**

cs.LG

11 pages, 8 figures, additional supplementary materials

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2309.03665v2) [paper-pdf](http://arxiv.org/pdf/2309.03665v2)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Ivan Y. Tyukin, Alexander N. Gorban, Alexander Bastounis, Desmond J. Higham

**Abstract**: Adversarial attacks dramatically change the output of an otherwise accurate learning system using a seemingly inconsequential modification to a piece of input data. Paradoxically, empirical evidence indicates that even systems which are robust to large random perturbations of the input data remain susceptible to small, easily constructed, adversarial perturbations of their inputs. Here, we show that this may be seen as a fundamental feature of classifiers working with high dimensional input data. We introduce a simple generic and generalisable framework for which key behaviours observed in practical systems arise with high probability -- notably the simultaneous susceptibility of the (otherwise accurate) model to easily constructed adversarial attacks, and robustness to random perturbations of the input data. We confirm that the same phenomena are directly observed in practical neural networks trained on standard image classification problems, where even large additive random noise fails to trigger the adversarial instability of the network. A surprising takeaway is that even small margins separating a classifier's decision surface from training and testing data can hide adversarial susceptibility from being detected using randomly sampled perturbations. Counterintuitively, using additive noise during training or testing is therefore inefficient for eradicating or detecting adversarial examples, and more demanding adversarial training is required.



## **46. Getting a-Round Guarantees: Floating-Point Attacks on Certified Robustness**

cs.CR

In Proceedings of the 2024 Workshop on Artificial Intelligence and  Security (AISec '24)

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2205.10159v5) [paper-pdf](http://arxiv.org/pdf/2205.10159v5)

**Authors**: Jiankai Jin, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstract**: Adversarial examples pose a security risk as they can alter decisions of a machine learning classifier through slight input perturbations. Certified robustness has been proposed as a mitigation where given an input $\mathbf{x}$, a classifier returns a prediction and a certified radius $R$ with a provable guarantee that any perturbation to $\mathbf{x}$ with $R$-bounded norm will not alter the classifier's prediction. In this work, we show that these guarantees can be invalidated due to limitations of floating-point representation that cause rounding errors. We design a rounding search method that can efficiently exploit this vulnerability to find adversarial examples against state-of-the-art certifications in two threat models, that differ in how the norm of the perturbation is computed. We show that the attack can be carried out against linear classifiers that have exact certifiable guarantees and against neural networks that have conservative certifications. In the weak threat model, our experiments demonstrate attack success rates over 50% on random linear classifiers, up to 23% on the MNIST dataset for linear SVM, and up to 15% for a neural network. In the strong threat model, the success rates are lower but positive. The floating-point errors exploited by our attacks can range from small to large (e.g., $10^{-13}$ to $10^{3}$) - showing that even negligible errors can be systematically exploited to invalidate guarantees provided by certified robustness. Finally, we propose a formal mitigation approach based on rounded interval arithmetic, encouraging future implementations of robustness certificates to account for limitations of modern computing architecture to provide sound certifiable guarantees.



## **47. Boosting Certificate Robustness for Time Series Classification with Efficient Self-Ensemble**

cs.LG

6 figures, 4 tables, 10 pages

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.02802v2) [paper-pdf](http://arxiv.org/pdf/2409.02802v2)

**Authors**: Chang Dong, Zhengyang Li, Liangwei Zheng, Weitong Chen, Wei Emma Zhang

**Abstract**: Recently, the issue of adversarial robustness in the time series domain has garnered significant attention. However, the available defense mechanisms remain limited, with adversarial training being the predominant approach, though it does not provide theoretical guarantees. Randomized Smoothing has emerged as a standout method due to its ability to certify a provable lower bound on robustness radius under $\ell_p$-ball attacks. Recognizing its success, research in the time series domain has started focusing on these aspects. However, existing research predominantly focuses on time series forecasting, or under the non-$\ell_p$ robustness in statistic feature augmentation for time series classification~(TSC). Our review found that Randomized Smoothing performs modestly in TSC, struggling to provide effective assurances on datasets with poor robustness. Therefore, we propose a self-ensemble method to enhance the lower bound of the probability confidence of predicted labels by reducing the variance of classification margins, thereby certifying a larger radius. This approach also addresses the computational overhead issue of Deep Ensemble~(DE) while remaining competitive and, in some cases, outperforming it in terms of robustness. Both theoretical analysis and experimental results validate the effectiveness of our method, demonstrating superior performance in robustness testing compared to baseline approaches.



## **48. A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems**

cs.RO

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2408.03515v2) [paper-pdf](http://arxiv.org/pdf/2408.03515v2)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Braunl, Jin B. Hong

**Abstract**: The integration of Large Language Models (LLMs) like GPT-4o into robotic systems represents a significant advancement in embodied artificial intelligence. These models can process multi-modal prompts, enabling them to generate more context-aware responses. However, this integration is not without challenges. One of the primary concerns is the potential security risks associated with using LLMs in robotic navigation tasks. These tasks require precise and reliable responses to ensure safe and effective operation. Multi-modal prompts, while enhancing the robot's understanding, also introduce complexities that can be exploited maliciously. For instance, adversarial inputs designed to mislead the model can lead to incorrect or dangerous navigational decisions. This study investigates the impact of prompt injections on mobile robot performance in LLM-integrated systems and explores secure prompt strategies to mitigate these risks. Our findings demonstrate a substantial overall improvement of approximately 30.8% in both attack detection and system performance with the implementation of robust defence mechanisms, highlighting their critical role in enhancing security and reliability in mission-oriented tasks.



## **49. Pseudorandom Permutations from Random Reversible Circuits**

cs.CC

v3: fixed minor errors

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2404.14648v3) [paper-pdf](http://arxiv.org/pdf/2404.14648v3)

**Authors**: William He, Ryan O'Donnell

**Abstract**: We study pseudorandomness properties of permutations on $\{0,1\}^n$ computed by random circuits made from reversible $3$-bit gates (permutations on $\{0,1\}^3$). Our main result is that a random circuit of depth $n \cdot \tilde{O}(k^2)$, with each layer consisting of $\approx n/3$ random gates in a fixed nearest-neighbor architecture, yields almost $k$-wise independent permutations. The main technical component is showing that the Markov chain on $k$-tuples of $n$-bit strings induced by a single random $3$-bit nearest-neighbor gate has spectral gap at least $1/n \cdot \tilde{O}(k)$. This improves on the original work of Gowers [Gowers96], who showed a gap of $1/\mathrm{poly}(n,k)$ for one random gate (with non-neighboring inputs); and, on subsequent work [HMMR05,BH08] improving the gap to $\Omega(1/n^2k)$ in the same setting.   From the perspective of cryptography, our result can be seen as a particularly simple/practical block cipher construction that gives provable statistical security against attackers with access to $k$~input-output pairs within few rounds. We also show that the Luby--Rackoff construction of pseudorandom permutations from pseudorandom functions can be implemented with reversible circuits. From this, we make progress on the complexity of the Minimum Reversible Circuit Size Problem (MRCSP), showing that block ciphers of fixed polynomial size are computationally secure against arbitrary polynomial-time adversaries, assuming the existence of one-way functions (OWFs).



## **50. PIP: Detecting Adversarial Examples in Large Vision-Language Models via Attention Patterns of Irrelevant Probe Questions**

cs.CV

Accepted by ACM Multimedia 2024 BNI track (Oral)

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2409.05076v1) [paper-pdf](http://arxiv.org/pdf/2409.05076v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Yu Wang

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated their powerful multimodal capabilities. However, they also face serious safety problems, as adversaries can induce robustness issues in LVLMs through the use of well-designed adversarial examples. Therefore, LVLMs are in urgent need of detection tools for adversarial examples to prevent incorrect responses. In this work, we first discover that LVLMs exhibit regular attention patterns for clean images when presented with probe questions. We propose an unconventional method named PIP, which utilizes the attention patterns of one randomly selected irrelevant probe question (e.g., "Is there a clock?") to distinguish adversarial examples from clean examples. Regardless of the image to be tested and its corresponding question, PIP only needs to perform one additional inference of the image to be tested and the probe question, and then achieves successful detection of adversarial examples. Even under black-box attacks and open dataset scenarios, our PIP, coupled with a simple SVM, still achieves more than 98% recall and a precision of over 90%. Our PIP is the first attempt to detect adversarial attacks on LVLMs via simple irrelevant probe questions, shedding light on deeper understanding and introspection within LVLMs. The code is available at https://github.com/btzyd/pip.



