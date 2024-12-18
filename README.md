# Latest Adversarial Attack Papers
**update at 2024-12-18 10:31:16**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Adaptive Epsilon Adversarial Training for Robust Gravitational Wave Parameter Estimation Using Normalizing Flows**

cs.LG

Due to new experimental results to add to the paper, this version no  longer accurately reflects the current state of our research. Therefore, we  are withdrawing the paper while further experiments are conducted. We will  submit a new version in the future. We apologize for any inconvenience this  may cause

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.07559v2) [paper-pdf](http://arxiv.org/pdf/2412.07559v2)

**Authors**: Yiqian Yang, Xihua Zhu, Fan Zhang

**Abstract**: Adversarial training with Normalizing Flow (NF) models is an emerging research area aimed at improving model robustness through adversarial samples. In this study, we focus on applying adversarial training to NF models for gravitational wave parameter estimation. We propose an adaptive epsilon method for Fast Gradient Sign Method (FGSM) adversarial training, which dynamically adjusts perturbation strengths based on gradient magnitudes using logarithmic scaling. Our hybrid architecture, combining ResNet and Inverse Autoregressive Flow, reduces the Negative Log Likelihood (NLL) loss by 47\% under FGSM attacks compared to the baseline model, while maintaining an NLL of 4.2 on clean data (only 5\% higher than the baseline). For perturbation strengths between 0.01 and 0.1, our model achieves an average NLL of 5.8, outperforming both fixed-epsilon (NLL: 6.7) and progressive-epsilon (NLL: 7.2) methods. Under stronger Projected Gradient Descent attacks with perturbation strength of 0.05, our model maintains an NLL of 6.4, demonstrating superior robustness while avoiding catastrophic overfitting.



## **2. PROSAC: Provably Safe Certification for Machine Learning Models under Adversarial Attacks**

cs.LG

Accepted to AAAI2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2402.02629v2) [paper-pdf](http://arxiv.org/pdf/2402.02629v2)

**Authors**: Chen Feng, Ziquan Liu, Zhuo Zhi, Ilija Bogunovic, Carsten Gerner-Beuerle, Miguel Rodrigues

**Abstract**: It is widely known that state-of-the-art machine learning models, including vision and language models, can be seriously compromised by adversarial perturbations. It is therefore increasingly relevant to develop capabilities to certify their performance in the presence of the most effective adversarial attacks. Our paper offers a new approach to certify the performance of machine learning models in the presence of adversarial attacks with population level risk guarantees. In particular, we introduce the notion of $(\alpha,\zeta)$-safe machine learning model. We propose a hypothesis testing procedure, based on the availability of a calibration set, to derive statistical guarantees providing that the probability of declaring that the adversarial (population) risk of a machine learning model is less than $\alpha$ (i.e. the model is safe), while the model is in fact unsafe (i.e. the model adversarial population risk is higher than $\alpha$), is less than $\zeta$. We also propose Bayesian optimization algorithms to determine efficiently whether a machine learning model is $(\alpha,\zeta)$-safe in the presence of an adversarial attack, along with statistical guarantees. We apply our framework to a range of machine learning models - including various sizes of vision Transformer (ViT) and ResNet models - impaired by a variety of adversarial attacks, such as PGDAttack, MomentumAttack, GenAttack and BanditAttack, to illustrate the operation of our approach. Importantly, we show that ViT's are generally more robust to adversarial attacks than ResNets, and large models are generally more robust than smaller models. Our approach goes beyond existing empirical adversarial risk-based certification guarantees. It formulates rigorous (and provable) performance guarantees that can be used to satisfy regulatory requirements mandating the use of state-of-the-art technical tools.



## **3. Deep Learning for Resilient Adversarial Decision Fusion in Byzantine Networks**

cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12739v1) [paper-pdf](http://arxiv.org/pdf/2412.12739v1)

**Authors**: Kassem Kallas

**Abstract**: This paper introduces a deep learning-based framework for resilient decision fusion in adversarial multi-sensor networks, providing a unified mathematical setup that encompasses diverse scenarios, including varying Byzantine node proportions, synchronized and unsynchronized attacks, unbalanced priors, adaptive strategies, and Markovian states. Unlike traditional methods, which depend on explicit parameter tuning and are limited by scenario-specific assumptions, the proposed approach employs a deep neural network trained on a globally constructed dataset to generalize across all cases without requiring adaptation. Extensive simulations validate the method's robustness, achieving superior accuracy, minimal error probability, and scalability compared to state-of-the-art techniques, while ensuring computational efficiency for real-time applications. This unified framework demonstrates the potential of deep learning to revolutionize decision fusion by addressing the challenges posed by Byzantine nodes in dynamic adversarial environments.



## **4. On the Impact of Hard Adversarial Instances on Overfitting in Adversarial Training**

cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2112.07324v2) [paper-pdf](http://arxiv.org/pdf/2112.07324v2)

**Authors**: Chen Liu, Zhichao Huang, Mathieu Salzmann, Tong Zhang, Sabine Süsstrunk

**Abstract**: Adversarial training is a popular method to robustify models against adversarial attacks. However, it exhibits much more severe overfitting than training on clean inputs. In this work, we investigate this phenomenon from the perspective of training instances, i.e., training input-target pairs. Based on a quantitative metric measuring the relative difficulty of an instance in the training set, we analyze the model's behavior on training instances of different difficulty levels. This lets us demonstrate that the decay in generalization performance of adversarial training is a result of fitting hard adversarial instances. We theoretically verify our observations for both linear and general nonlinear models, proving that models trained on hard instances have worse generalization performance than ones trained on easy instances, and that this generalization gap increases with the size of the adversarial budget. Finally, we investigate solutions to mitigate adversarial overfitting in several scenarios, including fast adversarial training and fine-tuning a pretrained model with additional data. Our results demonstrate that using training data adaptively improves the model's robustness.



## **5. Building Gradient Bridges: Label Leakage from Restricted Gradient Sharing in Federated Learning**

cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12640v1) [paper-pdf](http://arxiv.org/pdf/2412.12640v1)

**Authors**: Rui Zhang, Ka-Ho Chow, Ping Li

**Abstract**: The growing concern over data privacy, the benefits of utilizing data from diverse sources for model training, and the proliferation of networked devices with enhanced computational capabilities have all contributed to the rise of federated learning (FL). The clients in FL collaborate to train a global model by uploading gradients computed on their private datasets without collecting raw data. However, a new attack surface has emerged from gradient sharing, where adversaries can restore the label distribution of a victim's private data by analyzing the obtained gradients. To mitigate this privacy leakage, existing lightweight defenses restrict the sharing of gradients, such as encrypting the final-layer gradients or locally updating the parameters within. In this paper, we introduce a novel attack called Gradient Bridge (GDBR) that recovers the label distribution of training data from the limited gradient information shared in FL. GDBR explores the relationship between the layer-wise gradients, tracks the flow of gradients, and analytically derives the batch training labels. Extensive experiments show that GDBR can accurately recover more than 80% of labels in various FL settings. GDBR highlights the inadequacy of restricted gradient sharing-based defenses and calls for the design of effective defense schemes in FL.



## **6. Improving the Transferability of 3D Point Cloud Attack via Spectral-aware Admix and Optimization Designs**

cs.CV

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12626v1) [paper-pdf](http://arxiv.org/pdf/2412.12626v1)

**Authors**: Shiyu Hu, Daizong Liu, Wei Hu

**Abstract**: Deep learning models for point clouds have shown to be vulnerable to adversarial attacks, which have received increasing attention in various safety-critical applications such as autonomous driving, robotics, and surveillance. Existing 3D attackers generally design various attack strategies in the white-box setting, requiring the prior knowledge of 3D model details. However, real-world 3D applications are in the black-box setting, where we can only acquire the outputs of the target classifier. Although few recent works try to explore the black-box attack, they still achieve limited attack success rates (ASR). To alleviate this issue, this paper focuses on attacking the 3D models in a transfer-based black-box setting, where we first carefully design adversarial examples in a white-box surrogate model and then transfer them to attack other black-box victim models. Specifically, we propose a novel Spectral-aware Admix with Augmented Optimization method (SAAO) to improve the adversarial transferability. In particular, since traditional Admix strategy are deployed in the 2D domain that adds pixel-wise images for perturbing, we can not directly follow it to merge point clouds in coordinate domain as it will destroy the geometric shapes. Therefore, we design spectral-aware fusion that performs Graph Fourier Transform (GFT) to get spectral features of the point clouds and add them in the spectral domain. Afterward, we run a few steps with spectral-aware weighted Admix to select better optimization paths as well as to adjust corresponding learning weights. At last, we run more steps to generate adversarial spectral feature along the optimization path and perform Inverse-GFT on the adversarial spectral feature to obtain the adversarial example in the data domain. Experiments show that our SAAO achieves better transferability compared to existing 3D attack methods.



## **7. Jailbreaking? One Step Is Enough!**

cs.CL

17 pages

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12621v1) [paper-pdf](http://arxiv.org/pdf/2412.12621v1)

**Authors**: Weixiong Zheng, Peijian Zeng, Yiwei Li, Hongyan Wu, Nankai Lin, Junhao Chen, Aimin Yang, Yongmei Zhou

**Abstract**: Large language models (LLMs) excel in various tasks but remain vulnerable to jailbreak attacks, where adversaries manipulate prompts to generate harmful outputs. Examining jailbreak prompts helps uncover the shortcomings of LLMs. However, current jailbreak methods and the target model's defenses are engaged in an independent and adversarial process, resulting in the need for frequent attack iterations and redesigning attacks for different models. To address these gaps, we propose a Reverse Embedded Defense Attack (REDA) mechanism that disguises the attack intention as the "defense". intention against harmful content. Specifically, REDA starts from the target response, guiding the model to embed harmful content within its defensive measures, thereby relegating harmful content to a secondary role and making the model believe it is performing a defensive task. The attacking model considers that it is guiding the target model to deal with harmful content, while the target model thinks it is performing a defensive task, creating an illusion of cooperation between the two. Additionally, to enhance the model's confidence and guidance in "defensive" intentions, we adopt in-context learning (ICL) with a small number of attack examples and construct a corresponding dataset of attack examples. Extensive evaluations demonstrate that the REDA method enables cross-model attacks without the need to redesign attack strategies for different models, enables successful jailbreak in one iteration, and outperforms existing methods on both open-source and closed-source models.



## **8. WaterPark: A Robustness Assessment of Language Model Watermarking**

cs.CR

22 pages

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2411.13425v2) [paper-pdf](http://arxiv.org/pdf/2411.13425v2)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: Various watermarking methods (``watermarkers'') have been proposed to identify LLM-generated texts; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments? To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, by leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. We further explore the best practices to operate watermarkers in adversarial environments. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.



## **9. Attack On Prompt: Backdoor Attack in Prompt-Based Continual Learning**

cs.LG

Accepted to AAAI 2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2406.19753v2) [paper-pdf](http://arxiv.org/pdf/2406.19753v2)

**Authors**: Trang Nguyen, Anh Tran, Nhat Ho

**Abstract**: Prompt-based approaches offer a cutting-edge solution to data privacy issues in continual learning, particularly in scenarios involving multiple data suppliers where long-term storage of private user data is prohibited. Despite delivering state-of-the-art performance, its impressive remembering capability can become a double-edged sword, raising security concerns as it might inadvertently retain poisoned knowledge injected during learning from private user data. Following this insight, in this paper, we expose continual learning to a potential threat: backdoor attack, which drives the model to follow a desired adversarial target whenever a specific trigger is present while still performing normally on clean samples. We highlight three critical challenges in executing backdoor attacks on incremental learners and propose corresponding solutions: (1) \emph{Transferability}: We employ a surrogate dataset and manipulate prompt selection to transfer backdoor knowledge to data from other suppliers; (2) \emph{Resiliency}: We simulate static and dynamic states of the victim to ensure the backdoor trigger remains robust during intense incremental learning processes; and (3) \emph{Authenticity}: We apply binary cross-entropy loss as an anti-cheating factor to prevent the backdoor trigger from devolving into adversarial noise. Extensive experiments across various benchmark datasets and continual learners validate our continual backdoor framework, achieving up to $100\%$ attack success rate, with further ablation studies confirming our contributions' effectiveness.



## **10. Do Parameters Reveal More than Loss for Membership Inference?**

cs.LG

Accepted to Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2406.11544v3) [paper-pdf](http://arxiv.org/pdf/2406.11544v3)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks are used as a key tool for disclosure auditing. They aim to infer whether an individual record was used to train a model. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for stochastic gradient descent, and that optimal membership inference indeed requires white-box access. Our theoretical results lead to a new white-box inference attack, IHA (Inverse Hessian Attack), that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both auditors and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership inference.



## **11. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

cs.LG

accepted by KDD2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2408.08685v2) [paper-pdf](http://arxiv.org/pdf/2408.08685v2)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.



## **12. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

cs.CL

Review Version; Submitted to NAACL 2025 Demo Track

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12478v1) [paper-pdf](http://arxiv.org/pdf/2412.12478v1)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages.



## **13. Architectural Patterns for Designing Quantum Artificial Intelligence Systems**

cs.SE

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2411.10487v3) [paper-pdf](http://arxiv.org/pdf/2411.10487v3)

**Authors**: Mykhailo Klymenko, Thong Hoang, Xiwei Xu, Zhenchang Xing, Muhammad Usman, Qinghua Lu, Liming Zhu

**Abstract**: Utilising quantum computing technology to enhance artificial intelligence systems is expected to improve training and inference times, increase robustness against noise and adversarial attacks, and reduce the number of parameters without compromising accuracy. However, moving beyond proof-of-concept or simulations to develop practical applications of these systems while ensuring high software quality faces significant challenges due to the limitations of quantum hardware and the underdeveloped knowledge base in software engineering for such systems. In this work, we have conducted a systematic mapping study to identify the challenges and solutions associated with the software architecture of quantum-enhanced artificial intelligence systems. The results of the systematic mapping study reveal several architectural patterns that describe how quantum components can be integrated into inference engines, as well as middleware patterns that facilitate communication between classical and quantum components. Each pattern realises a trade-off between various software quality attributes, such as efficiency, scalability, trainability, simplicity, portability, and deployability. The outcomes of this work have been compiled into a catalogue of architectural patterns.



## **14. Adversarially robust generalization theory via Jacobian regularization for deep neural networks**

stat.ML

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12449v1) [paper-pdf](http://arxiv.org/pdf/2412.12449v1)

**Authors**: Dongya Wu, Xin Li

**Abstract**: Powerful deep neural networks are vulnerable to adversarial attacks. To obtain adversarially robust models, researchers have separately developed adversarial training and Jacobian regularization techniques. There are abundant theoretical and empirical studies for adversarial training, but theoretical foundations for Jacobian regularization are still lacking. In this study, we show that Jacobian regularization is closely related to adversarial training in that $\ell_{2}$ or $\ell_{1}$ Jacobian regularized loss serves as an approximate upper bound on the adversarially robust loss under $\ell_{2}$ or $\ell_{\infty}$ adversarial attack respectively. Further, we establish the robust generalization gap for Jacobian regularized risk minimizer via bounding the Rademacher complexity of both the standard loss function class and Jacobian regularization function class. Our theoretical results indicate that the norms of Jacobian are related to both standard and robust generalization. We also perform experiments on MNIST data classification to demonstrate that Jacobian regularized risk minimization indeed serves as a surrogate for adversarially robust risk minimization, and that reducing the norms of Jacobian can improve both standard and robust generalization. This study promotes both theoretical and empirical understandings to adversarially robust generalization via Jacobian regularization.



## **15. Quantum Adversarial Machine Learning and Defense Strategies: Challenges and Opportunities**

quant-ph

24 pages, 9 figures, 12 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.12373v1) [paper-pdf](http://arxiv.org/pdf/2412.12373v1)

**Authors**: Eric Yocam, Anthony Rizi, Mahesh Kamepalli, Varghese Vaidyan, Yong Wang, Gurcan Comert

**Abstract**: As quantum computing continues to advance, the development of quantum-secure neural networks is crucial to prevent adversarial attacks. This paper proposes three quantum-secure design principles: (1) using post-quantum cryptography, (2) employing quantum-resistant neural network architectures, and (3) ensuring transparent and accountable development and deployment. These principles are supported by various quantum strategies, including quantum data anonymization, quantum-resistant neural networks, and quantum encryption. The paper also identifies open issues in quantum security, privacy, and trust, and recommends exploring adaptive adversarial attacks and auto adversarial attacks as future directions. The proposed design principles and recommendations provide guidance for developing quantum-secure neural networks, ensuring the integrity and reliability of machine learning models in the quantum era.



## **16. Multi-Robot Target Tracking with Sensing and Communication Danger Zones**

cs.RO

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2404.07880v3) [paper-pdf](http://arxiv.org/pdf/2404.07880v3)

**Authors**: Jiazhen Liu, Peihan Li, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou

**Abstract**: Multi-robot target tracking finds extensive applications in different scenarios, such as environmental surveillance and wildfire management, which require the robustness of the practical deployment of multi-robot systems in uncertain and dangerous environments. Traditional approaches often focus on the performance of tracking accuracy with no modeling and assumption of the environments, neglecting potential environmental hazards which result in system failures in real-world deployments. To address this challenge, we investigate multi-robot target tracking in the adversarial environment considering sensing and communication attacks with uncertainty. We design specific strategies to avoid different danger zones and proposed a multi-agent tracking framework under the perilous environment. We approximate the probabilistic constraints and formulate practical optimization strategies to address computational challenges efficiently. We evaluate the performance of our proposed methods in simulations to demonstrate the ability of robots to adjust their risk-aware behaviors under different levels of environmental uncertainty and risk confidence. The proposed method is further validated via real-world robot experiments where a team of drones successfully track dynamic ground robots while being risk-aware of the sensing and/or communication danger zones.



## **17. Adversarial Attacks on Large Language Models in Medicine**

cs.AI

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2406.12259v3) [paper-pdf](http://arxiv.org/pdf/2406.12259v3)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.



## **18. Robust Synthetic Data-Driven Detection of Living-Off-the-Land Reverse Shells**

cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2402.18329v2) [paper-pdf](http://arxiv.org/pdf/2402.18329v2)

**Authors**: Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Fabio Roli

**Abstract**: Living-off-the-land (LOTL) techniques pose a significant challenge to security operations, exploiting legitimate tools to execute malicious commands that evade traditional detection methods. To address this, we present a robust augmentation framework for cyber defense systems as Security Information and Event Management (SIEM) solutions, enabling the detection of LOTL attacks such as reverse shells through machine learning. Leveraging real-world threat intelligence and adversarial training, our framework synthesizes diverse malicious datasets while preserving the variability of legitimate activity, ensuring high accuracy and low false-positive rates. We validate our approach through extensive experiments on enterprise-scale datasets, achieving a 90\% improvement in detection rates over non-augmented baselines at an industry-grade False Positive Rate (FPR) of $10^{-5}$. We define black-box data-driven attacks that successfully evade unprotected models, and develop defenses to mitigate them, producing adversarially robust variants of ML models. Ethical considerations are central to this work; we discuss safeguards for synthetic data generation and the responsible release of pre-trained models across four best performing architectures, including both adversarially and regularly trained variants: https://huggingface.co/dtrizna/quasarnix. Furthermore, we provide a malicious LOTL dataset containing over 1 million augmented attack variants to enable reproducible research and community collaboration: https://huggingface.co/datasets/dtrizna/QuasarNix. This work offers a reproducible, scalable, and production-ready defense against evolving LOTL threats.



## **19. Sonar-based Deep Learning in Underwater Robotics: Overview, Robustness and Challenges**

cs.RO

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11840v1) [paper-pdf](http://arxiv.org/pdf/2412.11840v1)

**Authors**: Martin Aubard, Ana Madureira, Luís Teixeira, José Pinto

**Abstract**: With the growing interest in underwater exploration and monitoring, Autonomous Underwater Vehicles (AUVs) have become essential. The recent interest in onboard Deep Learning (DL) has advanced real-time environmental interaction capabilities relying on efficient and accurate vision-based DL models. However, the predominant use of sonar in underwater environments, characterized by limited training data and inherent noise, poses challenges to model robustness. This autonomy improvement raises safety concerns for deploying such models during underwater operations, potentially leading to hazardous situations. This paper aims to provide the first comprehensive overview of sonar-based DL under the scope of robustness. It studies sonar-based DL perception task models, such as classification, object detection, segmentation, and SLAM. Furthermore, the paper systematizes sonar-based state-of-the-art datasets, simulators, and robustness methods such as neural network verification, out-of-distribution, and adversarial attacks. This paper highlights the lack of robustness in sonar-based DL research and suggests future research pathways, notably establishing a baseline sonar-based dataset and bridging the simulation-to-reality gap.



## **20. Transferable Adversarial Face Attack with Text Controlled Attribute**

cs.CV

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11735v1) [paper-pdf](http://arxiv.org/pdf/2412.11735v1)

**Authors**: Wenyun Li, Zheng Zhang, Xiangyuan Lan, Dongmei Jiang

**Abstract**: Traditional adversarial attacks typically produce adversarial examples under norm-constrained conditions, whereas unrestricted adversarial examples are free-form with semantically meaningful perturbations. Current unrestricted adversarial impersonation attacks exhibit limited control over adversarial face attributes and often suffer from low transferability. In this paper, we propose a novel Text Controlled Attribute Attack (TCA$^2$) to generate photorealistic adversarial impersonation faces guided by natural language. Specifically, the category-level personal softmax vector is employed to precisely guide the impersonation attacks. Additionally, we propose both data and model augmentation strategies to achieve transferable attacks on unknown target models. Finally, a generative model, \textit{i.e}, Style-GAN, is utilized to synthesize impersonated faces with desired attributes. Extensive experiments on two high-resolution face recognition datasets validate that our TCA$^2$ method can generate natural text-guided adversarial impersonation faces with high transferability. We also evaluate our method on real-world face recognition systems, \textit{i.e}, Face++ and Aliyun, further demonstrating the practical potential of our approach.



## **21. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2408.11749v2) [paper-pdf](http://arxiv.org/pdf/2408.11749v2)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.



## **22. Take Fake as Real: Realistic-like Robust Black-box Adversarial Attack to Evade AIGC Detection**

cs.CV

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.06727v2) [paper-pdf](http://arxiv.org/pdf/2412.06727v2)

**Authors**: Caiyun Xie, Dengpan Ye, Yunming Zhang, Long Tang, Yunna Lv, Jiacheng Deng, Jiawei Song

**Abstract**: The security of AI-generated content (AIGC) detection is crucial for ensuring multimedia content credibility. To enhance detector security, research on adversarial attacks has become essential. However, most existing adversarial attacks focus only on GAN-generated facial images detection, struggle to be effective on multi-class natural images and diffusion-based detectors, and exhibit poor invisibility. To fill this gap, we first conduct an in-depth analysis of the vulnerability of AIGC detectors and discover the feature that detectors vary in vulnerability to different post-processing. Then, considering that the detector is agnostic in real-world scenarios and given this discovery, we propose a Realistic-like Robust Black-box Adversarial attack (R$^2$BA) with post-processing fusion optimization. Unlike typical perturbations, R$^2$BA uses real-world post-processing, i.e., Gaussian blur, JPEG compression, Gaussian noise and light spot to generate adversarial examples. Specifically, we use a stochastic particle swarm algorithm with inertia decay to optimize post-processing fusion intensity and explore the detector's decision boundary. Guided by the detector's fake probability, R$^2$BA enhances/weakens the detector-vulnerable/detector-robust post-processing intensity to strike a balance between adversariality and invisibility. Extensive experiments on popular/commercial AIGC detectors and datasets demonstrate that R$^2$BA exhibits impressive anti-detection performance, excellent invisibility, and strong robustness in GAN-based and diffusion-based cases. Compared to state-of-the-art white-box and black-box attacks, R$^2$BA shows significant improvements of 15\%--72\% and 21\%--47\% in anti-detection performance under the original and robust scenario respectively, offering valuable insights for the security of AIGC detection in real-world applications.



## **23. PriPHiT: Privacy-Preserving Hierarchical Training of Deep Neural Networks**

cs.CV

21 pages, 19 figures, 11 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2408.05092v2) [paper-pdf](http://arxiv.org/pdf/2408.05092v2)

**Authors**: Yamin Sepehri, Pedram Pad, Pascal Frossard, L. Andrea Dunbar

**Abstract**: The training phase of deep neural networks requires substantial resources and as such is often performed on cloud servers. However, this raises privacy concerns when the training dataset contains sensitive content, e.g., facial or medical images. In this work, we propose a method to perform the training phase of a deep learning model on both an edge device and a cloud server that prevents sensitive content being transmitted to the cloud while retaining the desired information. The proposed privacy-preserving method uses adversarial early exits to suppress the sensitive content at the edge and transmits the task-relevant information to the cloud. This approach incorporates noise addition during the training phase to provide a differential privacy guarantee. We extensively test our method on different facial and medical datasets with diverse attributes using various deep learning architectures, showcasing its outstanding performance. We also demonstrate the effectiveness of privacy preservation through successful defenses against different white-box, deep and GAN-based reconstruction attacks. This approach is designed for resource-constrained edge devices, ensuring minimal memory usage and computational overhead.



## **24. Towards Adversarial Robustness of Model-Level Mixture-of-Experts Architectures for Semantic Segmentation**

cs.CV

Accepted for publication at ICMLA 2024

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11608v1) [paper-pdf](http://arxiv.org/pdf/2412.11608v1)

**Authors**: Svetlana Pavlitska, Enrico Eisen, J. Marius Zöllner

**Abstract**: Vulnerability to adversarial attacks is a well-known deficiency of deep neural networks. Larger networks are generally more robust, and ensembling is one method to increase adversarial robustness: each model's weaknesses are compensated by the strengths of others. While an ensemble uses a deterministic rule to combine model outputs, a mixture of experts (MoE) includes an additional learnable gating component that predicts weights for the outputs of the expert models, thus determining their contributions to the final prediction. MoEs have been shown to outperform ensembles on specific tasks, yet their susceptibility to adversarial attacks has not been studied yet. In this work, we evaluate the adversarial vulnerability of MoEs for semantic segmentation of urban and highway traffic scenes. We show that MoEs are, in most cases, more robust to per-instance and universal white-box adversarial attacks and can better withstand transfer attacks. Our code is available at \url{https://github.com/KASTEL-MobilityLab/mixtures-of-experts/}.



## **25. Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations**

cs.LG

Accepted by ICML2024

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2405.05075v2) [paper-pdf](http://arxiv.org/pdf/2405.05075v2)

**Authors**: Xuyang Zhong, Yixiao Huang, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations bounded by $l_0$ norm. We propose a white-box PGD-like attack method named sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against $l_0$ bounded adversarial perturbations. Moreover, the efficiency of sparse-PGD enables us to conduct adversarial training to build robust models against sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks. Codes are available at https://github.com/CityU-MLO/sPGD.



## **26. DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models**

cs.LG

Accepted by the Main Technical Track of the 39th Annual AAAI  Conference on Artificial Intelligence (AAAI-2025)

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.08160v3) [paper-pdf](http://arxiv.org/pdf/2412.08160v3)

**Authors**: Haonan Yuan, Qingyun Sun, Zhaonan Wang, Xingcheng Fu, Cheng Ji, Yongjian Wang, Bo Jin, Jianxin Li

**Abstract**: Dynamic graphs exhibit intertwined spatio-temporal evolutionary patterns, widely existing in the real world. Nevertheless, the structure incompleteness, noise, and redundancy result in poor robustness for Dynamic Graph Neural Networks (DGNNs). Dynamic Graph Structure Learning (DGSL) offers a promising way to optimize graph structures. However, aside from encountering unacceptable quadratic complexity, it overly relies on heuristic priors, making it hard to discover underlying predictive patterns. How to efficiently refine the dynamic structures, capture intrinsic dependencies, and learn robust representations, remains under-explored. In this work, we propose the novel DG-Mamba, a robust and efficient Dynamic Graph structure learning framework with the Selective State Space Models (Mamba). To accelerate the spatio-temporal structure learning, we propose a kernelized dynamic message-passing operator that reduces the quadratic time complexity to linear. To capture global intrinsic dynamics, we establish the dynamic graph as a self-contained system with State Space Model. By discretizing the system states with the cross-snapshot graph adjacency, we enable the long-distance dependencies capturing with the selective snapshot scan. To endow learned dynamic structures more expressive with informativeness, we propose the self-supervised Principle of Relevant Information for DGSL to regularize the most relevant yet least redundant information, enhancing global robustness. Extensive experiments demonstrate the superiority of the robustness and efficiency of our DG-Mamba compared with the state-of-the-art baselines against adversarial attacks.



## **27. Enhancing Robustness in Incremental Learning with Adversarial Training**

cs.CV

Accepted to AAAI 2025

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2312.03289v3) [paper-pdf](http://arxiv.org/pdf/2312.03289v3)

**Authors**: Seungju Cho, Hongsin Lee, Changick Kim

**Abstract**: Adversarial training is one of the most effective approaches against adversarial attacks. However, adversarial training has primarily been studied in scenarios where data for all classes is provided, with limited research conducted in the context of incremental learning where knowledge is introduced sequentially. In this study, we investigate Adversarially Robust Class Incremental Learning (ARCIL), which deals with adversarial robustness in incremental learning. We first explore a series of baselines that integrate incremental learning with existing adversarial training methods, finding that they lead to conflicts between acquiring new knowledge and retaining past knowledge. Furthermore, we discover that training new knowledge causes the disappearance of a key characteristic in robust models: a flat loss landscape in input space. To address such issues, we propose a novel and robust baseline for ARCIL, named \textbf{FL}atness-preserving \textbf{A}dversarial \textbf{I}ncremental learning for \textbf{R}obustness (\textbf{FLAIR}). Experimental results demonstrate that FLAIR significantly outperforms other baselines. To the best of our knowledge, we are the first to comprehensively investigate the baselines, challenges, and solutions for ARCIL, which we believe represents a significant advance toward achieving real-world robustness. Codes are available at \url{https://github.com/HongsinLee/FLAIR}.



## **28. UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models**

cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11441v1) [paper-pdf](http://arxiv.org/pdf/2412.11441v1)

**Authors**: Yuning Han, Bingyin Zhao, Rui Chu, Feng Luo, Biplab Sikdar, Yingjie Lao

**Abstract**: Recent studies show that diffusion models (DMs) are vulnerable to backdoor attacks. Existing backdoor attacks impose unconcealed triggers (e.g., a gray box and eyeglasses) that contain evident patterns, rendering remarkable attack effects yet easy detection upon human inspection and defensive algorithms. While it is possible to improve stealthiness by reducing the strength of the backdoor, doing so can significantly compromise its generality and effectiveness. In this paper, we propose UIBDiffusion, the universal imperceptible backdoor attack for diffusion models, which allows us to achieve superior attack and generation performance while evading state-of-the-art defenses. We propose a novel trigger generation approach based on universal adversarial perturbations (UAPs) and reveal that such perturbations, which are initially devised for fooling pre-trained discriminative models, can be adapted as potent imperceptible backdoor triggers for DMs. We evaluate UIBDiffusion on multiple types of DMs with different kinds of samplers across various datasets and targets. Experimental results demonstrate that UIBDiffusion brings three advantages: 1) Universality, the imperceptible trigger is universal (i.e., image and model agnostic) where a single trigger is effective to any images and all diffusion models with different samplers; 2) Utility, it achieves comparable generation quality (e.g., FID) and even better attack success rate (i.e., ASR) at low poison rates compared to the prior works; and 3) Undetectability, UIBDiffusion is plausible to human perception and can bypass Elijah and TERD, the SOTA defenses against backdoors for DMs. We will release our backdoor triggers and code.



## **29. Exploiting the Index Gradients for Optimization-Based Jailbreaking on Large Language Models**

cs.CL

13 pages,2 figures, accepted by COLING 2025

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.08615v2) [paper-pdf](http://arxiv.org/pdf/2412.08615v2)

**Authors**: Jiahui Li, Yongchang Hao, Haoyu Xu, Xing Wang, Yu Hong

**Abstract**: Despite the advancements in training Large Language Models (LLMs) with alignment techniques to enhance the safety of generated content, these models remain susceptible to jailbreak, an adversarial attack method that exposes security vulnerabilities in LLMs. Notably, the Greedy Coordinate Gradient (GCG) method has demonstrated the ability to automatically generate adversarial suffixes that jailbreak state-of-the-art LLMs. However, the optimization process involved in GCG is highly time-consuming, rendering the jailbreaking pipeline inefficient. In this paper, we investigate the process of GCG and identify an issue of Indirect Effect, the key bottleneck of the GCG optimization. To this end, we propose the Model Attack Gradient Index GCG (MAGIC), that addresses the Indirect Effect by exploiting the gradient information of the suffix tokens, thereby accelerating the procedure by having less computation and fewer iterations. Our experiments on AdvBench show that MAGIC achieves up to a 1.5x speedup, while maintaining Attack Success Rates (ASR) on par or even higher than other baselines. Our MAGIC achieved an ASR of 74% on the Llama-2 and an ASR of 54% when conducting transfer attacks on GPT-3.5. Code is available at https://github.com/jiah-li/magic.



## **30. Deep Learning Model Security: Threats and Defenses**

cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.08969v2) [paper-pdf](http://arxiv.org/pdf/2412.08969v2)

**Authors**: Tianyang Wang, Ziqian Bi, Yichao Zhang, Ming Liu, Weiche Hsieh, Pohsun Feng, Lawrence K. Q. Yan, Yizhu Wen, Benji Peng, Junyu Liu, Keyu Chen, Sen Zhang, Ming Li, Chuanqi Jiang, Xinyuan Song, Junjie Yang, Bowen Jing, Jintao Ren, Junhao Song, Hong-Ming Tseng, Silin Chen, Yunze Wang, Chia Xin Liang, Jiawei Xu, Xuanhe Pan, Jinlang Wang, Qian Niu

**Abstract**: Deep learning has transformed AI applications but faces critical security challenges, including adversarial attacks, data poisoning, model theft, and privacy leakage. This survey examines these vulnerabilities, detailing their mechanisms and impact on model integrity and confidentiality. Practical implementations, including adversarial examples, label flipping, and backdoor attacks, are explored alongside defenses such as adversarial training, differential privacy, and federated learning, highlighting their strengths and limitations.   Advanced methods like contrastive and self-supervised learning are presented for enhancing robustness. The survey concludes with future directions, emphasizing automated defenses, zero-trust architectures, and the security challenges of large AI models. A balanced approach to performance and security is essential for developing reliable deep learning systems.



## **31. A Comprehensive Review of Adversarial Attacks on Machine Learning**

cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11384v1) [paper-pdf](http://arxiv.org/pdf/2412.11384v1)

**Authors**: Syed Quiser Ahmed, Bharathi Vokkaliga Ganesh, Sathyanarayana Sampath Kumar, Prakhar Mishra, Ravi Anand, Bhanuteja Akurathi

**Abstract**: This research provides a comprehensive overview of adversarial attacks on AI and ML models, exploring various attack types, techniques, and their potential harms. We also delve into the business implications, mitigation strategies, and future research directions. To gain practical insights, we employ the Adversarial Robustness Toolbox (ART) [1] library to simulate these attacks on real-world use cases, such as self-driving cars. Our goal is to inform practitioners and researchers about the challenges and opportunities in defending AI systems against adversarial threats. By providing a comprehensive comparison of different attack methods, we aim to contribute to the development of more robust and secure AI systems.



## **32. Comprehensive Survey on Adversarial Examples in Cybersecurity: Impacts, Challenges, and Mitigation Strategies**

cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.12217v1) [paper-pdf](http://arxiv.org/pdf/2412.12217v1)

**Authors**: Li Li

**Abstract**: Deep learning (DL) has significantly transformed cybersecurity, enabling advancements in malware detection, botnet identification, intrusion detection, user authentication, and encrypted traffic analysis. However, the rise of adversarial examples (AE) poses a critical challenge to the robustness and reliability of DL-based systems. These subtle, crafted perturbations can deceive models, leading to severe consequences like misclassification and system vulnerabilities. This paper provides a comprehensive review of the impact of AE attacks on key cybersecurity applications, highlighting both their theoretical and practical implications. We systematically examine the methods used to generate adversarial examples, their specific effects across various domains, and the inherent trade-offs attackers face between efficacy and resource efficiency. Additionally, we explore recent advancements in defense mechanisms, including gradient masking, adversarial training, and detection techniques, evaluating their potential to enhance model resilience. By summarizing cutting-edge research, this study aims to bridge the gap between adversarial research and practical security applications, offering insights to fortify the adoption of DL solutions in cybersecurity.



## **33. Failures to Find Transferable Image Jailbreaks Between Vision-Language Models**

cs.CL

NeurIPS 2024 Workshops: RBFM (Best Paper), Frontiers in AdvML (Oral),  Red Teaming GenAI (Oral), SoLaR (Spotlight), SATA

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2407.15211v2) [paper-pdf](http://arxiv.org/pdf/2407.15211v2)

**Authors**: Rylan Schaeffer, Dan Valentine, Luke Bailey, James Chua, Cristóbal Eyzaguirre, Zane Durante, Joe Benton, Brando Miranda, Henry Sleight, John Hughes, Rajashree Agrawal, Mrinank Sharma, Scott Emmons, Sanmi Koyejo, Ethan Perez

**Abstract**: The integration of new modalities into frontier AI systems offers exciting capabilities, but also increases the possibility such systems can be adversarially manipulated in undesirable ways. In this work, we focus on a popular class of vision-language models (VLMs) that generate text outputs conditioned on visual and textual inputs. We conducted a large-scale empirical study to assess the transferability of gradient-based universal image ``jailbreaks" using a diverse set of over 40 open-parameter VLMs, including 18 new VLMs that we publicly release. Overall, we find that transferable gradient-based image jailbreaks are extremely difficult to obtain. When an image jailbreak is optimized against a single VLM or against an ensemble of VLMs, the jailbreak successfully jailbreaks the attacked VLM(s), but exhibits little-to-no transfer to any other VLMs; transfer is not affected by whether the attacked and target VLMs possess matching vision backbones or language models, whether the language model underwent instruction-following and/or safety-alignment training, or many other factors. Only two settings display partially successful transfer: between identically-pretrained and identically-initialized VLMs with slightly different VLM training data, and between different training checkpoints of a single VLM. Leveraging these results, we then demonstrate that transfer can be significantly improved against a specific target VLM by attacking larger ensembles of ``highly-similar" VLMs. These results stand in stark contrast to existing evidence of universal and transferable text jailbreaks against language models and transferable adversarial attacks against image classifiers, suggesting that VLMs may be more robust to gradient-based transfer attacks.



## **34. Dissecting Adversarial Robustness of Multimodal LM Agents**

cs.LG

Oral presentation at NeurIPS 2024 Open-World Agents Workshop

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2406.12814v2) [paper-pdf](http://arxiv.org/pdf/2406.12814v2)

**Authors**: Chen Henry Wu, Rishi Shah, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan

**Abstract**: As language models (LMs) are used to build autonomous agents in real environments, ensuring their adversarial robustness becomes a critical challenge. Unlike chatbots, agents are compound systems with multiple components, which existing LM safety evaluations do not adequately address. To bridge this gap, we manually create 200 targeted adversarial tasks and evaluation functions in a realistic threat model on top of VisualWebArena, a real environment for web-based agents. In order to systematically examine the robustness of various multimodal we agents, we propose the Agent Robustness Evaluation (ARE) framework. ARE views the agent as a graph showing the flow of intermediate outputs between components and decomposes robustness as the flow of adversarial information on the graph. First, we find that we can successfully break a range of the latest agents that use black-box frontier LLMs, including those that perform reflection and tree-search. With imperceptible perturbations to a single product image (less than 5% of total web page pixels), an attacker can hijack these agents to execute targeted adversarial goals with success rates up to 67%. We also use ARE to rigorously evaluate how the robustness changes as new components are added. We find that new components that typically improve benign performance can open up new vulnerabilities and harm robustness. An attacker can compromise the evaluator used by the reflexion agent and the value function of the tree search agent, which increases the attack success relatively by 15% and 20%. Our data and code for attacks, defenses, and evaluation are available at https://github.com/ChenWu98/agent-attack



## **35. Finding a Wolf in Sheep's Clothing: Combating Adversarial Text-To-Image Prompts with Text Summarization**

cs.CR

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.12212v1) [paper-pdf](http://arxiv.org/pdf/2412.12212v1)

**Authors**: Portia Cooper, Harshita Narnoli, Mihai Surdeanu

**Abstract**: Text-to-image models are vulnerable to the stepwise "Divide-and-Conquer Attack" (DACA) that utilize a large language model to obfuscate inappropriate content in prompts by wrapping sensitive text in a benign narrative. To mitigate stepwise DACA attacks, we propose a two-layer method involving text summarization followed by binary classification. We assembled the Adversarial Text-to-Image Prompt (ATTIP) dataset ($N=940$), which contained DACA-obfuscated and non-obfuscated prompts. From the ATTIP dataset, we created two summarized versions: one generated by a small encoder model and the other by a large language model. Then, we used an encoder classifier and a GPT-4o classifier to perform content moderation on the summarized and unsummarized prompts. When compared with a classifier that operated over the unsummarized data, our method improved F1 score performance by 31%. Further, the highest recorded F1 score achieved (98%) was produced by the encoder classifier on a summarized ATTIP variant. This study indicates that pre-classification text summarization can inoculate content detection models against stepwise DACA obfuscations.



## **36. Unpacking the Resilience of SNLI Contradiction Examples to Attacks**

cs.CL

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11172v1) [paper-pdf](http://arxiv.org/pdf/2412.11172v1)

**Authors**: Chetan Verma, Archit Agarwal

**Abstract**: Pre-trained models excel on NLI benchmarks like SNLI and MultiNLI, but their true language understanding remains uncertain. Models trained only on hypotheses and labels achieve high accuracy, indicating reliance on dataset biases and spurious correlations. To explore this issue, we applied the Universal Adversarial Attack to examine the model's vulnerabilities. Our analysis revealed substantial drops in accuracy for the entailment and neutral classes, whereas the contradiction class exhibited a smaller decline. Fine-tuning the model on an augmented dataset with adversarial examples restored its performance to near-baseline levels for both the standard and challenge sets. Our findings highlight the value of adversarial triggers in identifying spurious correlations and improving robustness while providing insights into the resilience of the contradiction class to adversarial attacks.



## **37. PGD-Imp: Rethinking and Unleashing Potential of Classic PGD with Dual Strategies for Imperceptible Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11168v1) [paper-pdf](http://arxiv.org/pdf/2412.11168v1)

**Authors**: Jin Li, Zitong Yu, Ziqiang He, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks have recently attracted increasing research interests. Existing methods typically incorporate external modules or loss terms other than a simple $l_p$-norm into the attack process to achieve imperceptibility, while we argue that such additional designs may not be necessary. In this paper, we rethink the essence of imperceptible attacks and propose two simple yet effective strategies to unleash the potential of PGD, the common and classical attack, for imperceptibility from an optimization perspective. Specifically, the Dynamic Step Size is introduced to find the optimal solution with minimal attack cost towards the decision boundary of the attacked model, and the Adaptive Early Stop strategy is adopted to reduce the redundant strength of adversarial perturbations to the minimum level. The proposed PGD-Imperceptible (PGD-Imp) attack achieves state-of-the-art results in imperceptible adversarial attacks for both untargeted and targeted scenarios. When performing untargeted attacks against ResNet-50, PGD-Imp attains 100$\%$ (+0.3$\%$) ASR, 0.89 (-1.76) $l_2$ distance, and 52.93 (+9.2) PSNR with 57s (-371s) running time, significantly outperforming existing methods.



## **38. The Superalignment of Superhuman Intelligence with Large Language Models**

cs.CL

Under review of Science China

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11145v1) [paper-pdf](http://arxiv.org/pdf/2412.11145v1)

**Authors**: Minlie Huang, Yingkang Wang, Shiyao Cui, Pei Ke, Jie Tang

**Abstract**: We have witnessed superhuman intelligence thanks to the fast development of large language models and multimodal language models. As the application of such superhuman models becomes more and more common, a critical question rises here: how can we ensure superhuman models are still safe, reliable and aligned well to human values? In this position paper, we discuss the concept of superalignment from the learning perspective to answer this question by outlining the learning paradigm shift from large-scale pretraining, supervised fine-tuning, to alignment training. We define superalignment as designing effective and efficient alignment algorithms to learn from noisy-labeled data (point-wise samples or pair-wise preference data) in a scalable way when the task becomes very complex for human experts to annotate and the model is stronger than human experts. We highlight some key research problems in superalignment, namely, weak-to-strong generalization, scalable oversight, and evaluation. We then present a conceptual framework for superalignment, which consists of three modules: an attacker which generates adversary queries trying to expose the weaknesses of a learner model; a learner which will refine itself by learning from scalable feedbacks generated by a critic model along with minimal human experts; and a critic which generates critics or explanations for a given query-response pair, with a target of improving the learner by criticizing. We discuss some important research problems in each component of this framework and highlight some interesting research ideas that are closely related to our proposed framework, for instance, self-alignment, self-play, self-refinement, and more. Last, we highlight some future research directions for superalignment, including identification of new emergent risks and multi-dimensional alignment.



## **39. Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models**

cs.CV

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2404.10335v4) [paper-pdf](http://arxiv.org/pdf/2404.10335v4)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo

**Abstract**: Adversarial attacks, particularly \textbf{targeted} transfer-based attacks, can be used to assess the adversarial robustness of large visual-language models (VLMs), allowing for a more thorough examination of potential security flaws before deployment. However, previous transfer-based adversarial attacks incur high costs due to high iteration counts and complex method structure. Furthermore, due to the unnaturalness of adversarial semantics, the generated adversarial examples have low transferability. These issues limit the utility of existing methods for assessing robustness. To address these issues, we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask method to disperse adversarial semantics throughout the image rather than concentrating them in a single area. Finally, AdvDiffVLM embeds more target semantics into adversarial examples after multiple iterations. Experimental results show that our method generates adversarial examples 5x to 10x faster than state-of-the-art transfer-based adversarial attacks while maintaining higher quality adversarial examples. Furthermore, compared to previous transfer-based adversarial attacks, the adversarial examples generated by our method have better transferability. Notably, AdvDiffVLM can successfully attack a variety of commercial VLMs in a black-box environment, including GPT-4V.



## **40. Impact of Adversarial Attacks on Deep Learning Model Explainability**

cs.LG

29 pages with reference included, submitted to a journal

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11119v1) [paper-pdf](http://arxiv.org/pdf/2412.11119v1)

**Authors**: Gazi Nazia Nur, Mohammad Ahnaf Sadat

**Abstract**: In this paper, we investigate the impact of adversarial attacks on the explainability of deep learning models, which are commonly criticized for their black-box nature despite their capacity for autonomous feature extraction. This black-box nature can affect the perceived trustworthiness of these models. To address this, explainability techniques such as GradCAM, SmoothGrad, and LIME have been developed to clarify model decision-making processes. Our research focuses on the robustness of these explanations when models are subjected to adversarial attacks, specifically those involving subtle image perturbations that are imperceptible to humans but can significantly mislead models. For this, we utilize attack methods like the Fast Gradient Sign Method (FGSM) and the Basic Iterative Method (BIM) and observe their effects on model accuracy and explanations. The results reveal a substantial decline in model accuracy, with accuracies dropping from 89.94% to 58.73% and 45.50% under FGSM and BIM attacks, respectively. Despite these declines in accuracy, the explanation of the models measured by metrics such as Intersection over Union (IoU) and Root Mean Square Error (RMSE) shows negligible changes, suggesting that these metrics may not be sensitive enough to detect the presence of adversarial perturbations.



## **41. Simulate and Eliminate: Revoke Backdoors for Generative Large Language Models**

cs.CR

To appear at AAAI 2025

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2405.07667v2) [paper-pdf](http://arxiv.org/pdf/2405.07667v2)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive data. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle scenarios where trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike other works that assume access to cleanly trained models, our safety-enhanced LLMs are able to revoke backdoors without any reference. Consequently, our safety-enhanced LLMs no longer produce targeted responses when the backdoor triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability.



## **42. Learning Robust and Privacy-Preserving Representations via Information Theory**

cs.LG

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11066v1) [paper-pdf](http://arxiv.org/pdf/2412.11066v1)

**Authors**: Binghui Zhang, Sayedeh Leila Noorbakhsh, Yun Dong, Yuan Hong, Binghui Wang

**Abstract**: Machine learning models are vulnerable to both security attacks (e.g., adversarial examples) and privacy attacks (e.g., private attribute inference). We take the first step to mitigate both the security and privacy attacks, and maintain task utility as well. Particularly, we propose an information-theoretic framework to achieve the goals through the lens of representation learning, i.e., learning representations that are robust to both adversarial examples and attribute inference adversaries. We also derive novel theoretical results under our framework, e.g., the inherent trade-off between adversarial robustness/utility and attribute privacy, and guaranteed attribute privacy leakage against attribute inference adversaries.



## **43. HTS-Attack: Heuristic Token Search for Jailbreaking Text-to-Image Models**

cs.CV

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2408.13896v3) [paper-pdf](http://arxiv.org/pdf/2408.13896v3)

**Authors**: Sensen Gao, Xiaojun Jia, Yihao Huang, Ranjie Duan, Jindong Gu, Yang Bai, Yang Liu, Qing Guo

**Abstract**: Text-to-Image(T2I) models have achieved remarkable success in image generation and editing, yet these models still have many potential issues, particularly in generating inappropriate or Not-Safe-For-Work(NSFW) content. Strengthening attacks and uncovering such vulnerabilities can advance the development of reliable and practical T2I models. Most of the previous works treat T2I models as white-box systems, using gradient optimization to generate adversarial prompts. However, accessing the model's gradient is often impossible in real-world scenarios. Moreover, existing defense methods, those using gradient masking, are designed to prevent attackers from obtaining accurate gradient information. While several black-box jailbreak attacks have been explored, they achieve the limited performance of jailbreaking T2I models due to difficulties associated with optimization in discrete spaces. To address this, we propose HTS-Attack, a heuristic token search attack method. HTS-Attack begins with an initialization that removes sensitive tokens, followed by a heuristic search where high-performing candidates are recombined and mutated. This process generates a new pool of candidates, and the optimal adversarial prompt is updated based on their effectiveness. By incorporating both optimal and suboptimal candidates, HTS-Attack avoids local optima and improves robustness in bypassing defenses. Extensive experiments validate the effectiveness of our method in attacking the latest prompt checkers, post-hoc image checkers, securely trained T2I models, and online commercial models.



## **44. Identification of Path Congestion Status for Network Performance Tomography using Deep Spatial-Temporal Learning**

cs.NI

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2412.10762v1) [paper-pdf](http://arxiv.org/pdf/2412.10762v1)

**Authors**: Chengze Du, Zhiwei Yu, Xiangyu Wang

**Abstract**: Network tomography plays a crucial role in assessing the operational status of internal links within networks through end-to-end path-level measurements, independently of cooperation from the network infrastructure. However, the accuracy of performance inference in internal network links heavily relies on comprehensive end-to-end path performance data. Most network tomography algorithms employ conventional threshold-based methods to identify congestion along paths, while these methods encounter limitations stemming from network complexities, resulting in inaccuracies such as misidentifying abnormal links and overlooking congestion attacks, thereby impeding algorithm performance. This paper introduces the concept of Additive Congestion Status to address these challenges effectively. Using a framework that combines Adversarial Autoencoders (AAE) with Long Short-Term Memory (LSTM) networks, this approach robustly categorizes (as uncongested, single-congested, or multiple-congested) and quantifies (regarding the number of congested links) the Additive Congestion Status. Leveraging prior path information and capturing spatio-temporal characteristics of probing flows, this method significantly enhances the localization of congested links and the inference of link performance compared to conventional network tomography algorithms, as demonstrated through experimental evaluations.



## **45. On Effects of Steering Latent Representation for Large Language Model Unlearning**

cs.CL

Accepted at AAAI-25 Main Technical Track

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2408.06223v2) [paper-pdf](http://arxiv.org/pdf/2408.06223v2)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. We investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. We show that RMU unlearned models are robust against adversarial jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU -- a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.



## **46. RAT: Adversarial Attacks on Deep Reinforcement Agents for Targeted Behaviors**

cs.LG

Accepted by AAAI 2025

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2412.10713v1) [paper-pdf](http://arxiv.org/pdf/2412.10713v1)

**Authors**: Fengshuo Bai, Runze Liu, Yali Du, Ying Wen, Yaodong Yang

**Abstract**: Evaluating deep reinforcement learning (DRL) agents against targeted behavior attacks is critical for assessing their robustness. These attacks aim to manipulate the victim into specific behaviors that align with the attacker's objectives, often bypassing traditional reward-based defenses. Prior methods have primarily focused on reducing cumulative rewards; however, rewards are typically too generic to capture complex safety requirements effectively. As a result, focusing solely on reward reduction can lead to suboptimal attack strategies, particularly in safety-critical scenarios where more precise behavior manipulation is needed. To address these challenges, we propose RAT, a method designed for universal, targeted behavior attacks. RAT trains an intention policy that is explicitly aligned with human preferences, serving as a precise behavioral target for the adversary. Concurrently, an adversary manipulates the victim's policy to follow this target behavior. To enhance the effectiveness of these attacks, RAT dynamically adjusts the state occupancy measure within the replay buffer, allowing for more controlled and effective behavior manipulation. Our empirical results on robotic simulation tasks demonstrate that RAT outperforms existing adversarial attack algorithms in inducing specific behaviors. Additionally, RAT shows promise in improving agent robustness, leading to more resilient policies. We further validate RAT by guiding Decision Transformer agents to adopt behaviors aligned with human preferences in various MuJoCo tasks, demonstrating its effectiveness across diverse tasks.



## **47. BinarySelect to Improve Accessibility of Black-Box Attack Research**

cs.CR

Accepted to COLING 2025, 17 pages, 5 figures, 11 tables

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.10617v1) [paper-pdf](http://arxiv.org/pdf/2412.10617v1)

**Authors**: Shatarupa Ghosh, Jonathan Rusert

**Abstract**: Adversarial text attack research is useful for testing the robustness of NLP models, however, the rise of transformers has greatly increased the time required to test attacks. Especially when researchers do not have access to adequate resources (e.g. GPUs). This can hinder attack research, as modifying one example for an attack can require hundreds of queries to a model, especially for black-box attacks. Often these attacks remove one token at a time to find the ideal one to change, requiring $n$ queries (the length of the text) right away. We propose a more efficient selection method called BinarySelect which combines binary search and attack selection methods to greatly reduce the number of queries needed to find a token. We find that BinarySelect only needs $\text{log}_2(n) * 2$ queries to find the first token compared to $n$ queries. We also test BinarySelect in an attack setting against 5 classifiers across 3 datasets and find a viable tradeoff between number of queries saved and attack effectiveness. For example, on the Yelp dataset, the number of queries is reduced by 32% (72 less) with a drop in attack effectiveness of only 5 points. We believe that BinarySelect can help future researchers study adversarial attacks and black-box problems more efficiently and opens the door for researchers with access to less resources.



## **48. Client-Side Patching against Backdoor Attacks in Federated Learning**

cs.CR

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.10605v1) [paper-pdf](http://arxiv.org/pdf/2412.10605v1)

**Authors**: Borja Molina Coronado

**Abstract**: Federated learning is a versatile framework for training models in decentralized environments. However, the trust placed in clients makes federated learning vulnerable to backdoor attacks launched by malicious participants. While many defenses have been proposed, they often fail short when facing heterogeneous data distributions among participating clients. In this paper, we propose a novel defense mechanism for federated learning systems designed to mitigate backdoor attacks on the clients-side. Our approach leverages adversarial learning techniques and model patching to neutralize the impact of backdoor attacks. Through extensive experiments on the MNIST and Fashion-MNIST datasets, we demonstrate that our defense effectively reduces backdoor accuracy, outperforming existing state-of-the-art defenses, such as LFighter, FLAME, and RoseAgg, in i.i.d. and non-i.i.d. scenarios, while maintaining competitive or superior accuracy on clean data.



## **49. Crosstalk-induced Side Channel Threats in Multi-Tenant NISQ Computers**

cs.ET

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.10507v1) [paper-pdf](http://arxiv.org/pdf/2412.10507v1)

**Authors**: Navnil Choudhury, Chaithanya Naik Mude, Sanjay Das, Preetham Chandra Tikkireddi, Swamit Tannu, Kanad Basu

**Abstract**: As quantum computing rapidly advances, its near-term applications are becoming increasingly evident. However, the high cost and under-utilization of quantum resources are prompting a shift from single-user to multi-user access models. In a multi-tenant environment, where multiple users share one quantum computer, protecting user confidentiality becomes crucial. The varied uses of quantum computers increase the risk that sensitive data encoded by one user could be compromised by others, rendering the protection of data integrity and confidentiality essential. In the evolving quantum computing landscape, it is imperative to study these security challenges within the scope of realistic threat model assumptions, wherein an adversarial user can mount practical attacks without relying on any heightened privileges afforded by physical access to a quantum computer or rogue cloud services. In this paper, we demonstrate the potential of crosstalk as an attack vector for the first time on a Noisy Intermediate Scale Quantum (NISQ) machine, that an adversarial user can exploit within a multi-tenant quantum computing model. The proposed side-channel attack is conducted with minimal and realistic adversarial privileges, with the overarching aim of uncovering the quantum algorithm being executed by a victim. Crosstalk signatures are used to estimate the presence of CNOT gates in the victim circuit, and subsequently, this information is encoded and classified by a graph-based learning model to identify the victim quantum algorithm. When evaluated on up to 336 benchmark circuits, our attack framework is found to be able to unveil the victim's quantum algorithm with up to 85.7\% accuracy.



## **50. MOREL: Enhancing Adversarial Robustness through Multi-Objective Representation Learning**

cs.LG

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2410.01697v3) [paper-pdf](http://arxiv.org/pdf/2410.01697v3)

**Authors**: Sedjro Salomon Hotegni, Sebastian Peitz

**Abstract**: Extensive research has shown that deep neural networks (DNNs) are vulnerable to slight adversarial perturbations$-$small changes to the input data that appear insignificant but cause the model to produce drastically different outputs. In addition to augmenting training data with adversarial examples generated from a specific attack method, most of the current defense strategies necessitate modifying the original model architecture components to improve robustness or performing test-time data purification to handle adversarial attacks. In this work, we demonstrate that strong feature representation learning during training can significantly enhance the original model's robustness. We propose MOREL, a multi-objective feature representation learning approach, encouraging classification models to produce similar features for inputs within the same class, despite perturbations. Our training method involves an embedding space where cosine similarity loss and multi-positive contrastive loss are used to align natural and adversarial features from the model encoder and ensure tight clustering. Concurrently, the classifier is motivated to achieve accurate predictions. Through extensive experiments, we demonstrate that our approach significantly enhances the robustness of DNNs against white-box and black-box adversarial attacks, outperforming other methods that similarly require no architectural changes or test-time data purification. Our code is available at https://github.com/salomonhotegni/MOREL



