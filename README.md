# Latest Adversarial Attack Papers
**update at 2023-07-24 11:33:42**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples**

cs.CL

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11729v1) [paper-pdf](http://arxiv.org/pdf/2307.11729v1)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, the effectiveness of these detectors in real-life situations, such as when students use LLMs for writing homework assignments (e.g., essays) and quickly learn how to evade these detectors, has not been explored. In this paper, we propose OUTFOX, a novel framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output and apply this to the domain of student essays. In our framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect. While the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Our experiments show that our proposed detector learned in-context from the attacker improves the detection performance on the attacked dataset by up to +41.3 point F1-score. While our proposed attacker can drastically degrade the performance of the detector by up to -57.0 point F1-score compared to the paraphrasing method.



## **2. (Ab)using Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs**

cs.CR

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.10490v2) [paper-pdf](http://arxiv.org/pdf/2307.10490v2)

**Authors**: Eugene Bagdasaryan, Tsung-Yin Hsieh, Ben Nassi, Vitaly Shmatikov

**Abstract**: We demonstrate how images and sounds can be used for indirect prompt and instruction injection in multi-modal LLMs. An attacker generates an adversarial perturbation corresponding to the prompt and blends it into an image or audio recording. When the user asks the (unmodified, benign) model about the perturbed image or audio, the perturbation steers the model to output the attacker-chosen text and/or make the subsequent dialog follow the attacker's instruction. We illustrate this attack with several proof-of-concept examples targeting LLaVa and PandaGPT.



## **3. Fast Adaptive Test-Time Defense with Robust Features**

cs.LG

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11672v1) [paper-pdf](http://arxiv.org/pdf/2307.11672v1)

**Authors**: Anurag Singh, Mahalakshmi Sabanayagam, Krikamol Muandet, Debarghya Ghoshdastidar

**Abstract**: Adaptive test-time defenses are used to improve the robustness of deep neural networks to adversarial examples. However, existing methods significantly increase the inference time due to additional optimization on the model parameters or the input at test time. In this work, we propose a novel adaptive test-time defense strategy that is easy to integrate with any existing (robust) training procedure without additional test-time computation. Based on the notion of robustness of features that we present, the key idea is to project the trained models to the most robust feature space, thereby reducing the vulnerability to adversarial attacks in non-robust directions. We theoretically show that the top eigenspace of the feature matrix are more robust for a generalized additive model and support our argument for a large width neural network with the Neural Tangent Kernel (NTK) equivalence. We conduct extensive experiments on CIFAR-10 and CIFAR-100 datasets for several robustness benchmarks, including the state-of-the-art methods in RobustBench, and observe that the proposed method outperforms existing adaptive test-time defenses at much lower computation costs.



## **4. Improving Viewpoint Robustness for Visual Recognition via Adversarial Training**

cs.CV

14 pages, 12 figures. arXiv admin note: substantial text overlap with  arXiv:2307.10235

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11528v1) [paper-pdf](http://arxiv.org/pdf/2307.11528v1)

**Authors**: Shouwei Ruan, Yinpeng Dong, Hang Su, Jianteng Peng, Ning Chen, Xingxing Wei

**Abstract**: Viewpoint invariance remains challenging for visual recognition in the 3D world, as altering the viewing directions can significantly impact predictions for the same object. While substantial efforts have been dedicated to making neural networks invariant to 2D image translations and rotations, viewpoint invariance is rarely investigated. Motivated by the success of adversarial training in enhancing model robustness, we propose Viewpoint-Invariant Adversarial Training (VIAT) to improve the viewpoint robustness of image classifiers. Regarding viewpoint transformation as an attack, we formulate VIAT as a minimax optimization problem, where the inner maximization characterizes diverse adversarial viewpoints by learning a Gaussian mixture distribution based on the proposed attack method GMVFool. The outer minimization obtains a viewpoint-invariant classifier by minimizing the expected loss over the worst-case viewpoint distributions that can share the same one for different objects within the same category. Based on GMVFool, we contribute a large-scale dataset called ImageNet-V+ to benchmark viewpoint robustness. Experimental results show that VIAT significantly improves the viewpoint robustness of various image classifiers based on the diversity of adversarial viewpoints generated by GMVFool. Furthermore, we propose ViewRS, a certified viewpoint robustness method that provides a certified radius and accuracy to demonstrate the effectiveness of VIAT from the theoretical perspective.



## **5. Improving Transferability of Adversarial Examples via Bayesian Attacks**

cs.LG

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11334v1) [paper-pdf](http://arxiv.org/pdf/2307.11334v1)

**Authors**: Qizhang Li, Yiwen Guo, Xiaochen Yang, Wangmeng Zuo, Hao Chen

**Abstract**: This paper presents a substantial extension of our work published at ICLR. Our ICLR work advocated for enhancing transferability in adversarial examples by incorporating a Bayesian formulation into model parameters, which effectively emulates the ensemble of infinitely many deep neural networks, while, in this paper, we introduce a novel extension by incorporating the Bayesian formulation into the model input as well, enabling the joint diversification of both the model input and model parameters. Our empirical findings demonstrate that: 1) the combination of Bayesian formulations for both the model input and model parameters yields significant improvements in transferability; 2) by introducing advanced approximations of the posterior distribution over the model input, adversarial transferability achieves further enhancement, surpassing all state-of-the-arts when attacking without model fine-tuning. Moreover, we propose a principled approach to fine-tune model parameters in such an extended Bayesian formulation. The derived optimization objective inherently encourages flat minima in the parameter space and input space. Extensive experiments demonstrate that our method achieves a new state-of-the-art on transfer-based attacks, improving the average success rate on ImageNet and CIFAR-10 by 19.14% and 2.08%, respectively, when comparing with our ICLR basic Bayesian method. We will make our code publicly available.



## **6. Epsilon*: Privacy Metric for Machine Learning Models**

cs.LG

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11280v1) [paper-pdf](http://arxiv.org/pdf/2307.11280v1)

**Authors**: Diana M. Negoescu, Humberto Gonzalez, Saad Eddin Al Orjany, Jilei Yang, Yuliia Lut, Rahul Tandra, Xiaowen Zhang, Xinyi Zheng, Zach Douglas, Vidita Nolkha, Parvez Ahammad, Gennady Samorodnitsky

**Abstract**: We introduce Epsilon*, a new privacy metric for measuring the privacy risk of a single model instance prior to, during, or after deployment of privacy mitigation strategies. The metric does not require access to the training data sampling or model training algorithm. Epsilon* is a function of true positive and false positive rates in a hypothesis test used by an adversary in a membership inference attack. We distinguish between quantifying the privacy loss of a trained model instance and quantifying the privacy loss of the training mechanism which produces this model instance. Existing approaches in the privacy auditing literature provide lower bounds for the latter, while our metric provides a lower bound for the former by relying on an (${\epsilon}$,${\delta}$)-type of quantification of the privacy of the trained model instance. We establish a relationship between these lower bounds and show how to implement Epsilon* to avoid numerical and noise amplification instability. We further show in experiments on benchmark public data sets that Epsilon* is sensitive to privacy risk mitigation by training with differential privacy (DP), where the value of Epsilon* is reduced by up to 800% compared to the Epsilon* values of non-DP trained baseline models. This metric allows privacy auditors to be independent of model owners, and enables all decision-makers to visualize the privacy-utility landscape to make informed decisions regarding the trade-offs between model privacy and utility.



## **7. Preprocessors Matter! Realistic Decision-Based Attacks on Machine Learning Systems**

cs.CR

ICML 2023. Code can be found at  https://github.com/google-research/preprocessor-aware-black-box-attack

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2210.03297v2) [paper-pdf](http://arxiv.org/pdf/2210.03297v2)

**Authors**: Chawin Sitawarin, Florian Tramèr, Nicholas Carlini

**Abstract**: Decision-based attacks construct adversarial examples against a machine learning (ML) model by making only hard-label queries. These attacks have mainly been applied directly to standalone neural networks. However, in practice, ML models are just one component of a larger learning system. We find that by adding a single preprocessor in front of a classifier, state-of-the-art query-based attacks are up to 7$\times$ less effective at attacking a prediction pipeline than at attacking the model alone. We explain this discrepancy by the fact that most preprocessors introduce some notion of invariance to the input space. Hence, attacks that are unaware of this invariance inevitably waste a large number of queries to re-discover or overcome it. We, therefore, develop techniques to (i) reverse-engineer the preprocessor and then (ii) use this extracted information to attack the end-to-end system. Our preprocessors extraction method requires only a few hundred queries, and our preprocessor-aware attacks recover the same efficacy as when attacking the model alone. The code can be found at https://github.com/google-research/preprocessor-aware-black-box-attack.



## **8. Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation**

eess.IV

This paper has been accepted in MICCAI 2023 conference

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.07269v2) [paper-pdf](http://arxiv.org/pdf/2307.07269v2)

**Authors**: Asif Hanif, Muzammal Naseer, Salman Khan, Mubarak Shah, Fahad Shahbaz Khan

**Abstract**: It is imperative to ensure the robustness of deep learning models in critical applications such as, healthcare. While recent advances in deep learning have improved the performance of volumetric medical image segmentation models, these models cannot be deployed for real-world applications immediately due to their vulnerability to adversarial attacks. We present a 3D frequency domain adversarial attack for volumetric medical image segmentation models and demonstrate its advantages over conventional input or voxel domain attacks. Using our proposed attack, we introduce a novel frequency domain adversarial training approach for optimizing a robust model against voxel and frequency domain attacks. Moreover, we propose frequency consistency loss to regulate our frequency domain adversarial training that achieves a better tradeoff between model's performance on clean and adversarial samples. Code is publicly available at https://github.com/asif-hanif/vafa.



## **9. PATROL: Privacy-Oriented Pruning for Collaborative Inference Against Model Inversion Attacks**

cs.LG

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10981v1) [paper-pdf](http://arxiv.org/pdf/2307.10981v1)

**Authors**: Shiwei Ding, Lan Zhang, Miao Pan, Xiaoyong Yuan

**Abstract**: Collaborative inference has been a promising solution to enable resource-constrained edge devices to perform inference using state-of-the-art deep neural networks (DNNs). In collaborative inference, the edge device first feeds the input to a partial DNN locally and then uploads the intermediate result to the cloud to complete the inference. However, recent research indicates model inversion attacks (MIAs) can reconstruct input data from intermediate results, posing serious privacy concerns for collaborative inference. Existing perturbation and cryptography techniques are inefficient and unreliable in defending against MIAs while performing accurate inference. This paper provides a viable solution, named PATROL, which develops privacy-oriented pruning to balance privacy, efficiency, and utility of collaborative inference. PATROL takes advantage of the fact that later layers in a DNN can extract more task-specific features. Given limited local resources for collaborative inference, PATROL intends to deploy more layers at the edge based on pruning techniques to enforce task-specific features for inference and reduce task-irrelevant but sensitive features for privacy preservation. To achieve privacy-oriented pruning, PATROL introduces two key components: Lipschitz regularization and adversarial reconstruction training, which increase the reconstruction errors by reducing the stability of MIAs and enhance the target inference model by adversarial training, respectively.



## **10. Risk-optimized Outlier Removal for Robust Point Cloud Classification**

cs.CV

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10875v1) [paper-pdf](http://arxiv.org/pdf/2307.10875v1)

**Authors**: Xinke Li, Junchi Lu

**Abstract**: The popularity of point cloud deep models for safety-critical purposes has increased, but the reliability and security of these models can be compromised by intentional or naturally occurring point cloud noise. To combat this issue, we present a novel point cloud outlier removal method called PointCVaR, which empowers standard-trained models to eliminate additional outliers and restore the data. Our approach begins by conducting attribution analysis to determine the influence of each point on the model output, which we refer to as point risk. We then optimize the process of filtering high-risk points using Conditional Value at Risk (CVaR) as the objective. The rationale for this approach is based on the observation that noise points in point clouds tend to cluster in the tail of the risk distribution, with a low frequency but a high level of risk, resulting in significant interference with classification results. Despite requiring no additional training effort, our method produces exceptional results in various removal-and-classification experiments for noisy point clouds, which are corrupted by random noise, adversarial noise, and backdoor trigger noise. Impressively, it achieves 87% accuracy in defense against the backdoor attack by removing triggers. Overall, the proposed PointCVaR effectively eliminates noise points and enhances point cloud classification, making it a promising plug-in module for various models in different scenarios.



## **11. Adversarial attacks for mixtures of classifiers**

cs.LG

7 pages + 4 pages of appendix. 5 figures in main text

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10788v1) [paper-pdf](http://arxiv.org/pdf/2307.10788v1)

**Authors**: Lucas Gnecco Heredia, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Mixtures of classifiers (a.k.a. randomized ensembles) have been proposed as a way to improve robustness against adversarial attacks. However, it has been shown that existing attacks are not well suited for this kind of classifiers. In this paper, we discuss the problem of attacking a mixture in a principled way and introduce two desirable properties of attacks based on a geometrical analysis of the problem (effectiveness and maximality). We then show that existing attacks do not meet both of these properties. Finally, we introduce a new attack called lattice climber attack with theoretical guarantees on the binary linear setting, and we demonstrate its performance by conducting experiments on synthetic and real datasets.



## **12. Friendly Noise against Adversarial Noise: A Powerful Defense against Data Poisoning Attacks**

cs.CR

Code available at: https://github.com/tianyu139/friendly-noise

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2208.10224v4) [paper-pdf](http://arxiv.org/pdf/2208.10224v4)

**Authors**: Tian Yu Liu, Yu Yang, Baharan Mirzasoleiman

**Abstract**: A powerful category of (invisible) data poisoning attacks modify a subset of training examples by small adversarial perturbations to change the prediction of certain test-time data. Existing defense mechanisms are not desirable to deploy in practice, as they often either drastically harm the generalization performance, or are attack-specific, and prohibitively slow to apply. Here, we propose a simple but highly effective approach that unlike existing methods breaks various types of invisible poisoning attacks with the slightest drop in the generalization performance. We make the key observation that attacks introduce local sharp regions of high training loss, which when minimized, results in learning the adversarial perturbations and makes the attack successful. To break poisoning attacks, our key idea is to alleviate the sharp loss regions introduced by poisons. To do so, our approach comprises two components: an optimized friendly noise that is generated to maximally perturb examples without degrading the performance, and a randomly varying noise component. The combination of both components builds a very light-weight but extremely effective defense against the most powerful triggerless targeted and hidden-trigger backdoor poisoning attacks, including Gradient Matching, Bulls-eye Polytope, and Sleeper Agent. We show that our friendly noise is transferable to other architectures, and adaptive attacks cannot break our defense due to its random noise component. Our code is available at: https://github.com/tianyu139/friendly-noise



## **13. A Holistic Assessment of the Reliability of Machine Learning Systems**

cs.LG

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10586v1) [paper-pdf](http://arxiv.org/pdf/2307.10586v1)

**Authors**: Anthony Corso, David Karamadian, Romeo Valentin, Mary Cooper, Mykel J. Kochenderfer

**Abstract**: As machine learning (ML) systems increasingly permeate high-stakes settings such as healthcare, transportation, military, and national security, concerns regarding their reliability have emerged. Despite notable progress, the performance of these systems can significantly diminish due to adversarial attacks or environmental changes, leading to overconfident predictions, failures to detect input faults, and an inability to generalize in unexpected scenarios. This paper proposes a holistic assessment methodology for the reliability of ML systems. Our framework evaluates five key properties: in-distribution accuracy, distribution-shift robustness, adversarial robustness, calibration, and out-of-distribution detection. A reliability score is also introduced and used to assess the overall system reliability. To provide insights into the performance of different algorithmic approaches, we identify and categorize state-of-the-art techniques, then evaluate a selection on real-world tasks using our proposed reliability metrics and reliability score. Our analysis of over 500 models reveals that designing for one metric does not necessarily constrain others but certain algorithmic techniques can improve reliability across multiple metrics simultaneously. This study contributes to a more comprehensive understanding of ML reliability and provides a roadmap for future research and development.



## **14. FACADE: A Framework for Adversarial Circuit Anomaly Detection and Evaluation**

cs.LG

Accepted as BlueSky Poster at 2023 ICML AdvML Workshop

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10563v1) [paper-pdf](http://arxiv.org/pdf/2307.10563v1)

**Authors**: Dhruv Pai, Andres Carranza, Rylan Schaeffer, Arnuv Tandon, Sanmi Koyejo

**Abstract**: We present FACADE, a novel probabilistic and geometric framework designed for unsupervised mechanistic anomaly detection in deep neural networks. Its primary goal is advancing the understanding and mitigation of adversarial attacks. FACADE aims to generate probabilistic distributions over circuits, which provide critical insights to their contribution to changes in the manifold properties of pseudo-classes, or high-dimensional modes in activation space, yielding a powerful tool for uncovering and combating adversarial attacks. Our approach seeks to improve model robustness, enhance scalable model oversight, and demonstrates promising applications in real-world deployment settings.



## **15. Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples**

cs.LG

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10562v1) [paper-pdf](http://arxiv.org/pdf/2307.10562v1)

**Authors**: Shaokui Wei, Mingda Zhang, Hongyuan Zha, Baoyuan Wu

**Abstract**: Backdoor attacks are serious security threats to machine learning models where an adversary can inject poisoned samples into the training set, causing a backdoored model which predicts poisoned samples with particular triggers to particular target classes, while behaving normally on benign samples. In this paper, we explore the task of purifying a backdoored model using a small clean dataset. By establishing the connection between backdoor risk and adversarial risk, we derive a novel upper bound for backdoor risk, which mainly captures the risk on the shared adversarial examples (SAEs) between the backdoored model and the purified model. This upper bound further suggests a novel bi-level optimization problem for mitigating backdoor using adversarial training techniques. To solve it, we propose Shared Adversarial Unlearning (SAU). Specifically, SAU first generates SAEs, and then, unlearns the generated SAEs such that they are either correctly classified by the purified model and/or differently classified by the two models, such that the backdoor effect in the backdoored model will be mitigated in the purified model. Experiments on various benchmark datasets and network architectures show that our proposed method achieves state-of-the-art performance for backdoor defense.



## **16. It Is All About Data: A Survey on the Effects of Data on Adversarial Robustness**

cs.LG

51 pages, 25 figures, under review

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2303.09767v2) [paper-pdf](http://arxiv.org/pdf/2303.09767v2)

**Authors**: Peiyu Xiong, Michael Tegegn, Jaskeerat Singh Sarin, Shubhraneel Pal, Julia Rubin

**Abstract**: Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to confuse the model into making a mistake. Such examples pose a serious threat to the applicability of machine-learning-based systems, especially in life- and safety-critical domains. To address this problem, the area of adversarial robustness investigates mechanisms behind adversarial attacks and defenses against these attacks. This survey reviews a particular subset of this literature that focuses on investigating properties of training data in the context of model robustness under evasion attacks. It first summarizes the main properties of data leading to adversarial vulnerability. It then discusses guidelines and techniques for improving adversarial robustness by enhancing the data representation and learning procedures, as well as techniques for estimating robustness guarantees given particular data. Finally, it discusses gaps of knowledge and promising future research directions in this area.



## **17. MultiRobustBench: Benchmarking Robustness Against Multiple Attacks**

cs.LG

ICML 2023

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2302.10980v3) [paper-pdf](http://arxiv.org/pdf/2302.10980v3)

**Authors**: Sihui Dai, Saeed Mahloujifar, Chong Xiang, Vikash Sehwag, Pin-Yu Chen, Prateek Mittal

**Abstract**: The bulk of existing research in defending against adversarial examples focuses on defending against a single (typically bounded Lp-norm) attack, but for a practical setting, machine learning (ML) models should be robust to a wide variety of attacks. In this paper, we present the first unified framework for considering multiple attacks against ML models. Our framework is able to model different levels of learner's knowledge about the test-time adversary, allowing us to model robustness against unforeseen attacks and robustness against unions of attacks. Using our framework, we present the first leaderboard, MultiRobustBench, for benchmarking multiattack evaluation which captures performance across attack types and attack strengths. We evaluate the performance of 16 defended models for robustness against a set of 9 different attack types, including Lp-based threat models, spatial transformations, and color changes, at 20 different attack strengths (180 attacks total). Additionally, we analyze the state of current defenses against multiple attacks. Our analysis shows that while existing defenses have made progress in terms of average robustness across the set of attacks used, robustness against the worst-case attack is still a big open problem as all existing models perform worse than random guessing.



## **18. Invariant Aggregator for Defending against Federated Backdoor Attacks**

cs.LG

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2210.01834v2) [paper-pdf](http://arxiv.org/pdf/2210.01834v2)

**Authors**: Xiaoyang Wang, Dimitrios Dimitriadis, Sanmi Koyejo, Shruti Tople

**Abstract**: Federated learning is gaining popularity as it enables training high-utility models across several clients without directly sharing their private data. As a downside, the federated setting makes the model vulnerable to various adversarial attacks in the presence of malicious clients. Despite the theoretical and empirical success in defending against attacks that aim to degrade models' utility, defense against backdoor attacks that increase model accuracy on backdoor samples exclusively without hurting the utility on other samples remains challenging. To this end, we first analyze the vulnerability of federated learning to backdoor attacks over a flat loss landscape which is common for well-designed neural networks such as Resnet [He et al., 2015] but is often overlooked by previous works. Over a flat loss landscape, misleading federated learning models to exclusively benefit malicious clients with backdoor samples do not require a significant difference between malicious and benign client-wise updates, making existing defenses insufficient. In contrast, we propose an invariant aggregator that redirects the aggregated update to invariant directions that are generally useful via selectively masking out the gradient elements that favor few and possibly malicious clients regardless of the difference magnitude. Theoretical results suggest that our approach provably mitigates backdoor attacks over both flat and sharp loss landscapes. Empirical results on three datasets with different modalities and varying numbers of clients further demonstrate that our approach mitigates a broad class of backdoor attacks with a negligible cost on the model utility.



## **19. Rethinking Backdoor Attacks**

cs.CR

ICML 2023

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2307.10163v1) [paper-pdf](http://arxiv.org/pdf/2307.10163v1)

**Authors**: Alaa Khaddaj, Guillaume Leclerc, Aleksandar Makelov, Kristian Georgiev, Hadi Salman, Andrew Ilyas, Aleksander Madry

**Abstract**: In a backdoor attack, an adversary inserts maliciously constructed backdoor examples into a training set to make the resulting model vulnerable to manipulation. Defending against such attacks typically involves viewing these inserted examples as outliers in the training set and using techniques from robust statistics to detect and remove them.   In this work, we present a different approach to the backdoor attack problem. Specifically, we show that without structural information about the training data distribution, backdoor attacks are indistinguishable from naturally-occurring features in the data--and thus impossible to "detect" in a general sense. Then, guided by this observation, we revisit existing defenses against backdoor attacks and characterize the (often latent) assumptions they make and on which they depend. Finally, we explore an alternative perspective on backdoor attacks: one that assumes these attacks correspond to the strongest feature in the training data. Under this assumption (which we make formal) we develop a new primitive for detecting backdoor attacks. Our primitive naturally gives rise to a detection algorithm that comes with theoretical guarantees and is effective in practice.



## **20. I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models**

cs.CV

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2306.07591v3) [paper-pdf](http://arxiv.org/pdf/2306.07591v3)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Modern image-to-text systems typically adopt the encoder-decoder framework, which comprises two main components: an image encoder, responsible for extracting image features, and a transformer-based decoder, used for generating captions. Taking inspiration from the analysis of neural networks' robustness against adversarial perturbations, we propose a novel gray-box algorithm for creating adversarial examples in image-to-text models. Unlike image classification tasks that have a finite set of class labels, finding visually similar adversarial examples in an image-to-text task poses greater challenges because the captioning system allows for a virtually infinite space of possible captions. In this paper, we present a gray-box adversarial attack on image-to-text, both untargeted and targeted. We formulate the process of discovering adversarial perturbations as an optimization problem that uses only the image-encoder component, meaning the proposed attack is language-model agnostic. Through experiments conducted on the ViT-GPT2 model, which is the most-used image-to-text model in Hugging Face, and the Flickr30k dataset, we demonstrate that our proposed attack successfully generates visually similar adversarial examples, both with untargeted and targeted captions. Notably, our attack operates in a gray-box manner, requiring no knowledge about the decoder module. We also show that our attacks fool the popular open-source platform Hugging Face.



## **21. Why Does Little Robustness Help? Understanding Adversarial Transferability From Surrogate Training**

cs.LG

Accepted by IEEE Symposium on Security and Privacy (Oakland) 2024; 21  pages, 12 figures, 13 tables

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2307.07873v2) [paper-pdf](http://arxiv.org/pdf/2307.07873v2)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.



## **22. Fix your downsampling ASAP! Be natively more robust via Aliasing and Spectral Artifact free Pooling**

cs.CV

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2307.09804v1) [paper-pdf](http://arxiv.org/pdf/2307.09804v1)

**Authors**: Julia Grabinski, Janis Keuper, Margret Keuper

**Abstract**: Convolutional neural networks encode images through a sequence of convolutions, normalizations and non-linearities as well as downsampling operations into potentially strong semantic embeddings. Yet, previous work showed that even slight mistakes during sampling, leading to aliasing, can be directly attributed to the networks' lack in robustness. To address such issues and facilitate simpler and faster adversarial training, [12] recently proposed FLC pooling, a method for provably alias-free downsampling - in theory. In this work, we conduct a further analysis through the lens of signal processing and find that such current pooling methods, which address aliasing in the frequency domain, are still prone to spectral leakage artifacts. Hence, we propose aliasing and spectral artifact-free pooling, short ASAP. While only introducing a few modifications to FLC pooling, networks using ASAP as downsampling method exhibit higher native robustness against common corruptions, a property that FLC pooling was missing. ASAP also increases native robustness against adversarial attacks on high and low resolution data while maintaining similar clean accuracy or even outperforming the baseline.



## **23. Improving the Transferability of Adversarial Attacks on Face Recognition with Beneficial Perturbation Feature Augmentation**

cs.CV

\c{opyright} 2023 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2210.16117v4) [paper-pdf](http://arxiv.org/pdf/2210.16117v4)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Zongyi Li, Ping Li

**Abstract**: Face recognition (FR) models can be easily fooled by adversarial examples, which are crafted by adding imperceptible perturbations on benign face images. The existence of adversarial face examples poses a great threat to the security of society. In order to build a more sustainable digital nation, in this paper, we improve the transferability of adversarial face examples to expose more blind spots of existing FR models. Though generating hard samples has shown its effectiveness in improving the generalization of models in training tasks, the effectiveness of utilizing this idea to improve the transferability of adversarial face examples remains unexplored. To this end, based on the property of hard samples and the symmetry between training tasks and adversarial attack tasks, we propose the concept of hard models, which have similar effects as hard samples for adversarial attack tasks. Utilizing the concept of hard models, we propose a novel attack method called Beneficial Perturbation Feature Augmentation Attack (BPFA), which reduces the overfitting of adversarial examples to surrogate FR models by constantly generating new hard models to craft the adversarial examples. Specifically, in the backpropagation, BPFA records the gradients on pre-selected feature maps and uses the gradient on the input image to craft the adversarial example. In the next forward propagation, BPFA leverages the recorded gradients to add beneficial perturbations on their corresponding feature maps to increase the loss. Extensive experiments demonstrate that BPFA can significantly boost the transferability of adversarial attacks on FR.



## **24. Making Substitute Models More Bayesian Can Enhance Transferability of Adversarial Examples**

cs.LG

Accepted by ICLR 2023, fix typos

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2302.05086v3) [paper-pdf](http://arxiv.org/pdf/2302.05086v3)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: The transferability of adversarial examples across deep neural networks (DNNs) is the crux of many black-box attacks. Many prior efforts have been devoted to improving the transferability via increasing the diversity in inputs of some substitute models. In this paper, by contrast, we opt for the diversity in substitute models and advocate to attack a Bayesian model for achieving desirable transferability. Deriving from the Bayesian formulation, we develop a principled strategy for possible finetuning, which can be combined with many off-the-shelf Gaussian posterior approximations over DNN parameters. Extensive experiments have been conducted to verify the effectiveness of our method, on common benchmark datasets, and the results demonstrate that our method outperforms recent state-of-the-arts by large margins (roughly 19% absolute increase in average attack success rate on ImageNet), and, by combining with these recent methods, further performance gain can be obtained. Our code: https://github.com/qizhangli/MoreBayesian-attack.



## **25. Reinforcing POD based model reduction techniques in reaction-diffusion complex networks using stochastic filtering and pattern recognition**

cs.CE

19 pages, 6 figures

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2307.09762v1) [paper-pdf](http://arxiv.org/pdf/2307.09762v1)

**Authors**: Abhishek Ajayakumar, Soumyendu Raha

**Abstract**: Complex networks are used to model many real-world systems. However, the dimensionality of these systems can make them challenging to analyze. Dimensionality reduction techniques like POD can be used in such cases. However, these models are susceptible to perturbations in the input data. We propose an algorithmic framework that combines techniques from pattern recognition (PR) and stochastic filtering theory to enhance the output of such models. The results of our study show that our method can improve the accuracy of the surrogate model under perturbed inputs. Deep Neural Networks (DNNs) are susceptible to adversarial attacks. However, recent research has revealed that neural Ordinary Differential Equations (ODEs) exhibit robustness in specific applications. We benchmark our algorithmic framework with a Neural ODE-based approach as a reference.



## **26. Unified Adversarial Patch for Cross-modal Attacks in the Physical World**

cs.CV

10 pages, 8 figures, accepted by ICCV2023

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2307.07859v2) [paper-pdf](http://arxiv.org/pdf/2307.07859v2)

**Authors**: Xingxing Wei, Yao Huang, Yitong Sun, Jie Yu

**Abstract**: Recently, physical adversarial attacks have been presented to evade DNNs-based object detectors. To ensure the security, many scenarios are simultaneously deployed with visible sensors and infrared sensors, leading to the failures of these single-modal physical attacks. To show the potential risks under such scenes, we propose a unified adversarial patch to perform cross-modal physical attacks, i.e., fooling visible and infrared object detectors at the same time via a single patch. Considering different imaging mechanisms of visible and infrared sensors, our work focuses on modeling the shapes of adversarial patches, which can be captured in different modalities when they change. To this end, we design a novel boundary-limited shape optimization to achieve the compact and smooth shapes, and thus they can be easily implemented in the physical world. In addition, to balance the fooling degree between visible detector and infrared detector during the optimization process, we propose a score-aware iterative evaluation, which can guide the adversarial patch to iteratively reduce the predicted scores of the multi-modal sensors. We finally test our method against the one-stage detector: YOLOv3 and the two-stage detector: Faster RCNN. Results show that our unified patch achieves an Attack Success Rate (ASR) of 73.33% and 69.17%, respectively. More importantly, we verify the effective attacks in the physical world when visible and infrared sensors shoot the objects under various settings like different angles, distances, postures, and scenes.



## **27. VISER: A Tractable Solution Concept for Games with Information Asymmetry**

cs.GT

17 pages, 6 figures

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.09652v1) [paper-pdf](http://arxiv.org/pdf/2307.09652v1)

**Authors**: Jeremy McMahan, Young Wu, Yudong Chen, Xiaojin Zhu, Qiaomin Xie

**Abstract**: Many real-world games suffer from information asymmetry: one player is only aware of their own payoffs while the other player has the full game information. Examples include the critical domain of security games and adversarial multi-agent reinforcement learning. Information asymmetry renders traditional solution concepts such as Strong Stackelberg Equilibrium (SSE) and Robust-Optimization Equilibrium (ROE) inoperative. We propose a novel solution concept called VISER (Victim Is Secure, Exploiter best-Responds). VISER enables an external observer to predict the outcome of such games. In particular, for security applications, VISER allows the victim to better defend itself while characterizing the most damaging attacks available to the attacker. We show that each player's VISER strategy can be computed independently in polynomial time using linear programming (LP). We also extend VISER to its Markov-perfect counterpart for Markov games, which can be solved efficiently using a series of LPs.



## **28. Dead Man's PLC: Towards Viable Cyber Extortion for Operational Technology**

cs.CR

13 pages, 19 figures

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.09549v1) [paper-pdf](http://arxiv.org/pdf/2307.09549v1)

**Authors**: Richard Derbyshire, Benjamin Green, Charl van der Walt, David Hutchison

**Abstract**: For decades, operational technology (OT) has enjoyed the luxury of being suitably inaccessible so as to experience directly targeted cyber attacks from only the most advanced and well-resourced adversaries. However, security via obscurity cannot last forever, and indeed a shift is happening whereby less advanced adversaries are showing an appetite for targeting OT. With this shift in adversary demographics, there will likely also be a shift in attack goals, from clandestine process degradation and espionage to overt cyber extortion (Cy-X). The consensus from OT cyber security practitioners suggests that, even if encryption-based Cy-X techniques were launched against OT assets, typical recovery practices designed for engineering processes would provide adequate resilience. In response, this paper introduces Dead Man's PLC (DM-PLC), a pragmatic step towards viable OT Cy-X that acknowledges and weaponises the resilience processes typically encountered. Using only existing functionality, DM-PLC considers an entire environment as the entity under ransom, whereby all assets constantly poll one another to ensure the attack remains untampered, treating any deviations as a detonation trigger akin to a Dead Man's switch. A proof of concept of DM-PLC is implemented and evaluated on an academically peer reviewed and industry validated OT testbed to demonstrate its malicious efficacy.



## **29. Untargeted Near-collision Attacks in Biometric Recognition**

cs.CR

Addition of results and correction of typos

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2304.01580v3) [paper-pdf](http://arxiv.org/pdf/2304.01580v3)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Kevin Thiry-Atighehchi

**Abstract**: A biometric recognition system can operate in two distinct modes, identification or verification. In the first mode, the system recognizes an individual by searching the enrolled templates of all the users for a match. In the second mode, the system validates a user's identity claim by comparing the fresh provided template with the enrolled template. The biometric transformation schemes usually produce binary templates that are better handled by cryptographic schemes, and the comparison is based on a distance that leaks information about the similarities between two biometric templates. Both the experimentally determined false match rate and false non-match rate through recognition threshold adjustment define the recognition accuracy, and hence the security of the system. To the best of our knowledge, few works provide a formal treatment of the security under minimum leakage of information, i.e., the binary outcome of a comparison with a threshold. In this paper, we rely on probabilistic modelling to quantify the security strength of binary templates. We investigate the influence of template size, database size and threshold on the probability of having a near-collision. We highlight several untargeted attacks on biometric systems considering naive and adaptive adversaries. Interestingly, these attacks can be launched both online and offline and, both in the identification mode and in the verification mode. We discuss the choice of parameters through the generic presented attacks.



## **30. Mitigating Intersection Attacks in Anonymous Microblogging**

cs.CR

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.09069v1) [paper-pdf](http://arxiv.org/pdf/2307.09069v1)

**Authors**: Sarah Abdelwahab Gaballah, Thanh Hoang Long Nguyen, Lamya Abdullah, Ephraim Zimmer, Max Mühlhäuser

**Abstract**: Anonymous microblogging systems are known to be vulnerable to intersection attacks due to network churn. An adversary that monitors all communications can leverage the churn to learn who is publishing what with increasing confidence over time. In this paper, we propose a protocol for mitigating intersection attacks in anonymous microblogging systems by grouping users into anonymity sets based on similarities in their publishing behavior. The protocol provides a configurable communication schedule for users in each set to manage the inevitable trade-off between latency and bandwidth overhead. In our evaluation, we use real-world datasets from two popular microblogging platforms, Twitter and Reddit, to simulate user publishing behavior. The results demonstrate that the protocol can protect users against intersection attacks at low bandwidth overhead when the users adhere to communication schedules. In addition, the protocol can sustain a slow degradation in the size of the anonymity set over time under various churn rates.



## **31. FedDefender: Client-Side Attack-Tolerant Federated Learning**

cs.CR

KDD'23 research track accepted

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.09048v1) [paper-pdf](http://arxiv.org/pdf/2307.09048v1)

**Authors**: Sungwon Park, Sungwon Han, Fangzhao Wu, Sundong Kim, Bin Zhu, Xing Xie, Meeyoung Cha

**Abstract**: Federated learning enables learning from decentralized data sources without compromising privacy, which makes it a crucial technique. However, it is vulnerable to model poisoning attacks, where malicious clients interfere with the training process. Previous defense mechanisms have focused on the server-side by using careful model aggregation, but this may not be effective when the data is not identically distributed or when attackers can access the information of benign clients. In this paper, we propose a new defense mechanism that focuses on the client-side, called FedDefender, to help benign clients train robust local models and avoid the adverse impact of malicious model updates from attackers, even when a server-side defense cannot identify or remove adversaries. Our method consists of two main components: (1) attack-tolerant local meta update and (2) attack-tolerant global knowledge distillation. These components are used to find noise-resilient model parameters while accurately extracting knowledge from a potentially corrupted global model. Our client-side defense strategy has a flexible structure and can work in conjunction with any existing server-side strategies. Evaluations of real-world scenarios across multiple datasets show that the proposed method enhances the robustness of federated learning against model poisoning attacks.



## **32. Discretization-based ensemble model for robust learning in IoT**

cs.LG

15 pages

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.08955v1) [paper-pdf](http://arxiv.org/pdf/2307.08955v1)

**Authors**: Anahita Namvar, Chandra Thapa, Salil S. Kanhere

**Abstract**: IoT device identification is the process of recognizing and verifying connected IoT devices to the network. This is an essential process for ensuring that only authorized devices can access the network, and it is necessary for network management and maintenance. In recent years, machine learning models have been used widely for automating the process of identifying devices in the network. However, these models are vulnerable to adversarial attacks that can compromise their accuracy and effectiveness. To better secure device identification models, discretization techniques enable reduction in the sensitivity of machine learning models to adversarial attacks contributing to the stability and reliability of the model. On the other hand, Ensemble methods combine multiple heterogeneous models to reduce the impact of remaining noise or errors in the model. Therefore, in this paper, we integrate discretization techniques and ensemble methods and examine it on model robustness against adversarial attacks. In other words, we propose a discretization-based ensemble stacking technique to improve the security of our ML models. We evaluate the performance of different ML-based IoT device identification models against white box and black box attacks using a real-world dataset comprised of network traffic from 28 IoT devices. We demonstrate that the proposed method enables robustness to the models for IoT device identification.



## **33. On the Robustness of Split Learning against Adversarial Attacks**

cs.LG

accepted by ECAI 2023, camera-ready version

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.07916v2) [paper-pdf](http://arxiv.org/pdf/2307.07916v2)

**Authors**: Mingyuan Fan, Cen Chen, Chengyu Wang, Wenmeng Zhou, Jun Huang

**Abstract**: Split learning enables collaborative deep learning model training while preserving data privacy and model security by avoiding direct sharing of raw data and model details (i.e., sever and clients only hold partial sub-networks and exchange intermediate computations). However, existing research has mainly focused on examining its reliability for privacy protection, with little investigation into model security. Specifically, by exploring full models, attackers can launch adversarial attacks, and split learning can mitigate this severe threat by only disclosing part of models to untrusted servers.This paper aims to evaluate the robustness of split learning against adversarial attacks, particularly in the most challenging setting where untrusted servers only have access to the intermediate layers of the model.Existing adversarial attacks mostly focus on the centralized setting instead of the collaborative setting, thus, to better evaluate the robustness of split learning, we develop a tailored attack called SPADV, which comprises two stages: 1) shadow model training that addresses the issue of lacking part of the model and 2) local adversarial attack that produces adversarial examples to evaluate.The first stage only requires a few unlabeled non-IID data, and, in the second stage, SPADV perturbs the intermediate output of natural samples to craft the adversarial ones. The overall cost of the proposed attack process is relatively low, yet the empirical attack effectiveness is significantly high, demonstrating the surprising vulnerability of split learning to adversarial attacks.



## **34. Hiding Visual Information via Obfuscating Adversarial Perturbations**

cs.CV

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2209.15304v3) [paper-pdf](http://arxiv.org/pdf/2209.15304v3)

**Authors**: Zhigang Su, Dawei Zhou, Decheng Liu, Nannan Wang, Zhen Wang, Xinbo Gao

**Abstract**: Growing leakage and misuse of visual information raise security and privacy concerns, which promotes the development of information protection. Existing adversarial perturbations-based methods mainly focus on the de-identification against deep learning models. However, the inherent visual information of the data has not been well protected. In this work, inspired by the Type-I adversarial attack, we propose an adversarial visual information hiding method to protect the visual privacy of data. Specifically, the method generates obfuscating adversarial perturbations to obscure the visual information of the data. Meanwhile, it maintains the hidden objectives to be correctly predicted by models. In addition, our method does not modify the parameters of the applied model, which makes it flexible for different scenarios. Experimental results on the recognition and classification tasks demonstrate that the proposed method can effectively hide visual information and hardly affect the performances of models. The code is available in the supplementary material.



## **35. Unstoppable Attack: Label-Only Model Inversion via Conditional Diffusion Model**

cs.AI

11 pages, 6 figures, 2 tables

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.08424v2) [paper-pdf](http://arxiv.org/pdf/2307.08424v2)

**Authors**: Rongke Liu

**Abstract**: Model inversion attacks (MIAs) are aimed at recovering private data from a target model's training set, which poses a threat to the privacy of deep learning models. MIAs primarily focus on the white-box scenario where the attacker has full access to the structure and parameters of the target model. However, practical applications are black-box, it is not easy for adversaries to obtain model-related parameters, and various models only output predicted labels. Existing black-box MIAs primarily focused on designing the optimization strategy, and the generative model is only migrated from the GAN used in white-box MIA. Our research is the pioneering study of feasible attack models in label-only black-box scenarios, to the best of our knowledge.   In this paper, we develop a novel method of MIA using the conditional diffusion model to recover the precise sample of the target without any extra optimization, as long as the target model outputs the label. Two primary techniques are introduced to execute the attack. Firstly, select an auxiliary dataset that is relevant to the target model task, and the labels predicted by the target model are used as conditions to guide the training process. Secondly, target labels and random standard normally distributed noise are input into the trained conditional diffusion model, generating target samples with pre-defined guidance strength. We then filter out the most robust and representative samples. Furthermore, we propose for the first time to use Learned Perceptual Image Patch Similarity (LPIPS) as one of the evaluation metrics for MIA, with systematic quantitative and qualitative evaluation in terms of attack accuracy, realism, and similarity. Experimental results show that this method can generate similar and accurate data to the target without optimization and outperforms generators of previous approaches in the label-only scenario.



## **36. Exploring the Unprecedented Privacy Risks of the Metaverse**

cs.CR

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2207.13176v3) [paper-pdf](http://arxiv.org/pdf/2207.13176v3)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song

**Abstract**: Thirty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Behind the scenes, an adversarial program had accurately inferred over 25 personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender, within just a few minutes of gameplay. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. While virtual telepresence applications (and the so-called "metaverse") have recently received increased attention and investment from major tech firms, these environments remain relatively under-studied from a security and privacy standpoint. In this work, we illustrate how VR attackers can covertly ascertain dozens of personal data attributes from seemingly-anonymous users of popular metaverse applications like VRChat. These attackers can be as simple as other VR users without special privilege, and the potential scale and scope of this data collection far exceed what is feasible within traditional mobile and web applications. We aim to shed light on the unique privacy risks of the metaverse, and provide the first holistic framework for understanding intrusive data harvesting attacks in these emerging VR ecosystems.



## **37. TorMult: Introducing a Novel Tor Bandwidth Inflation Attack**

cs.CR

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.08550v1) [paper-pdf](http://arxiv.org/pdf/2307.08550v1)

**Authors**: Christoph Sendner, Jasper Stang, Alexandra Dmitrienko, Raveen Wijewickrama, Murtuza Jadliwala

**Abstract**: The Tor network is the most prominent system for providing anonymous communication to web users, with a daily user base of 2 million users. However, since its inception, it has been constantly targeted by various traffic fingerprinting and correlation attacks aiming at deanonymizing its users. A critical requirement for these attacks is to attract as much user traffic to adversarial relays as possible, which is typically accomplished by means of bandwidth inflation attacks. This paper proposes a new inflation attack vector in Tor, referred to as TorMult, which enables inflation of measured bandwidth. The underlying attack technique exploits resource sharing among Tor relay nodes and employs a cluster of attacker-controlled relays with coordinated resource allocation within the cluster to deceive bandwidth measurers into believing that each relay node in the cluster possesses ample resources. We propose two attack variants, C-TorMult and D-TorMult, and test both versions in a private Tor test network. Our evaluation demonstrates that an attacker can inflate the measured bandwidth by a factor close to n using C-TorMult and nearly half n*N using D-TorMult, where n is the size of the cluster hosted on one server and N is the number of servers. Furthermore, our theoretical analysis reveals that gaining control over half of the Tor network's traffic can be achieved by employing just 10 dedicated servers with a cluster size of 109 relays running the TorMult attack, each with a bandwidth of 100MB/s. The problem is further exacerbated by the fact that Tor not only allows resource sharing but, according to recent reports, even promotes it.



## **38. FocusedCleaner: Sanitizing Poisoned Graphs for Robust GNN-based Node Classification**

cs.LG

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2210.13815v2) [paper-pdf](http://arxiv.org/pdf/2210.13815v2)

**Authors**: Yulin Zhu, Liang Tong, Gaolei Li, Xiapu Luo, Kai Zhou

**Abstract**: Graph Neural Networks (GNNs) are vulnerable to data poisoning attacks, which will generate a poisoned graph as the input to the GNN models. We present FocusedCleaner as a poisoned graph sanitizer to effectively identify the poison injected by attackers. Specifically, FocusedCleaner provides a sanitation framework consisting of two modules: bi-level structural learning and victim node detection. In particular, the structural learning module will reverse the attack process to steadily sanitize the graph while the detection module provides ``the focus" -- a narrowed and more accurate search region -- to structural learning. These two modules will operate in iterations and reinforce each other to sanitize a poisoned graph step by step. As an important application, we show that the adversarial robustness of GNNs trained over the sanitized graph for the node classification task is significantly improved. Extensive experiments demonstrate that FocusedCleaner outperforms the state-of-the-art baselines both on poisoned graph sanitation and improving robustness.



## **39. Tracking Fringe and Coordinated Activity on Twitter Leading Up To the US Capitol Attack**

cs.SI

11 pages (including references), 8 figures, 1 table. Accepted at The  18th International AAAI Conference on Web and Social Media

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2302.04450v2) [paper-pdf](http://arxiv.org/pdf/2302.04450v2)

**Authors**: Vishnuprasad Padinjaredath Suresh, Gianluca Nogara, Felipe Cardoso, Stefano Cresci, Silvia Giordano, Luca Luceri

**Abstract**: The aftermath of the 2020 US Presidential Election witnessed an unprecedented attack on the democratic values of the country through the violent insurrection at Capitol Hill on January 6th, 2021. The attack was fueled by the proliferation of conspiracy theories and misleading claims about the integrity of the election pushed by political elites and fringe communities on social media. In this study, we explore the evolution of fringe content and conspiracy theories on Twitter in the seven months leading up to the Capitol attack. We examine the suspicious coordinated activity carried out by users sharing fringe content, finding evidence of common adversarial manipulation techniques ranging from targeted amplification to manufactured consensus. Further, we map out the temporal evolution of, and the relationship between, fringe and conspiracy theories, which eventually coalesced into the rhetoric of a stolen election, with the hashtag #stopthesteal, alongside QAnon-related narratives. Our findings further highlight how social media platforms offer fertile ground for the widespread proliferation of conspiracies during major societal events, which can potentially lead to offline coordinated actions and organized violence.



## **40. Adversarial Self-Attack Defense and Spatial-Temporal Relation Mining for Visible-Infrared Video Person Re-Identification**

cs.CV

11 pages,8 figures

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.03903v2) [paper-pdf](http://arxiv.org/pdf/2307.03903v2)

**Authors**: Huafeng Li, Le Xu, Yafei Zhang, Dapeng Tao, Zhengtao Yu

**Abstract**: In visible-infrared video person re-identification (re-ID), extracting features not affected by complex scenes (such as modality, camera views, pedestrian pose, background, etc.) changes, and mining and utilizing motion information are the keys to solving cross-modal pedestrian identity matching. To this end, the paper proposes a new visible-infrared video person re-ID method from a novel perspective, i.e., adversarial self-attack defense and spatial-temporal relation mining. In this work, the changes of views, posture, background and modal discrepancy are considered as the main factors that cause the perturbations of person identity features. Such interference information contained in the training samples is used as an adversarial perturbation. It performs adversarial attacks on the re-ID model during the training to make the model more robust to these unfavorable factors. The attack from the adversarial perturbation is introduced by activating the interference information contained in the input samples without generating adversarial samples, and it can be thus called adversarial self-attack. This design allows adversarial attack and defense to be integrated into one framework. This paper further proposes a spatial-temporal information-guided feature representation network to use the information in video sequences. The network cannot only extract the information contained in the video-frame sequences but also use the relation of the local information in space to guide the network to extract more robust features. The proposed method exhibits compelling performance on large-scale cross-modality video datasets. The source code of the proposed method will be released at https://github.com/lhf12278/xxx.



## **41. A Machine Learning based Empirical Evaluation of Cyber Threat Actors High Level Attack Patterns over Low level Attack Patterns in Attributing Attacks**

cs.CR

20 pages

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.10252v1) [paper-pdf](http://arxiv.org/pdf/2307.10252v1)

**Authors**: Umara Noor, Sawera Shahid, Rimsha Kanwal, Zahid Rashid

**Abstract**: Cyber threat attribution is the process of identifying the actor of an attack incident in cyberspace. An accurate and timely threat attribution plays an important role in deterring future attacks by applying appropriate and timely defense mechanisms. Manual analysis of attack patterns gathered by honeypot deployments, intrusion detection systems, firewalls, and via trace-back procedures is still the preferred method of security analysts for cyber threat attribution. Such attack patterns are low-level Indicators of Compromise (IOC). They represent Tactics, Techniques, Procedures (TTP), and software tools used by the adversaries in their campaigns. The adversaries rarely re-use them. They can also be manipulated, resulting in false and unfair attribution. To empirically evaluate and compare the effectiveness of both kinds of IOC, there are two problems that need to be addressed. The first problem is that in recent research works, the ineffectiveness of low-level IOC for cyber threat attribution has been discussed intuitively. An empirical evaluation for the measure of the effectiveness of low-level IOC based on a real-world dataset is missing. The second problem is that the available dataset for high-level IOC has a single instance for each predictive class label that cannot be used directly for training machine learning models. To address these problems in this research work, we empirically evaluate the effectiveness of low-level IOC based on a real-world dataset that is specifically built for comparative analysis with high-level IOC. The experimental results show that the high-level IOC trained models effectively attribute cyberattacks with an accuracy of 95% as compared to the low-level IOC trained models where accuracy is 40%.



## **42. Analyzing the Impact of Adversarial Examples on Explainable Machine Learning**

cs.LG

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.08327v1) [paper-pdf](http://arxiv.org/pdf/2307.08327v1)

**Authors**: Prathyusha Devabhakthini, Sasmita Parida, Raj Mani Shukla, Suvendu Chandan Nayak

**Abstract**: Adversarial attacks are a type of attack on machine learning models where an attacker deliberately modifies the inputs to cause the model to make incorrect predictions. Adversarial attacks can have serious consequences, particularly in applications such as autonomous vehicles, medical diagnosis, and security systems. Work on the vulnerability of deep learning models to adversarial attacks has shown that it is very easy to make samples that make a model predict things that it doesn't want to. In this work, we analyze the impact of model interpretability due to adversarial attacks on text classification problems. We develop an ML-based classification model for text data. Then, we introduce the adversarial perturbations on the text data to understand the classification performance after the attack. Subsequently, we analyze and interpret the model's explainability before and after the attack



## **43. Adversarial Attacks on Traffic Sign Recognition: A Survey**

cs.CV

Accepted for publication at ICECCME2023

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.08278v1) [paper-pdf](http://arxiv.org/pdf/2307.08278v1)

**Authors**: Svetlana Pavlitska, Nico Lambing, J. Marius Zöllner

**Abstract**: Traffic sign recognition is an essential component of perception in autonomous vehicles, which is currently performed almost exclusively with deep neural networks (DNNs). However, DNNs are known to be vulnerable to adversarial attacks. Several previous works have demonstrated the feasibility of adversarial attacks on traffic sign recognition models. Traffic signs are particularly promising for adversarial attack research due to the ease of performing real-world attacks using printed signs or stickers. In this work, we survey existing works performing either digital or real-world attacks on traffic sign detection and classification models. We provide an overview of the latest advancements and highlight the existing research areas that require further investigation.



## **44. Towards Stealthy Backdoor Attacks against Speech Recognition via Elements of Sound**

cs.SD

13 pages

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.08208v1) [paper-pdf](http://arxiv.org/pdf/2307.08208v1)

**Authors**: Hanbo Cai, Pengcheng Zhang, Hai Dong, Yan Xiao, Stefanos Koffas, Yiming Li

**Abstract**: Deep neural networks (DNNs) have been widely and successfully adopted and deployed in various applications of speech recognition. Recently, a few works revealed that these models are vulnerable to backdoor attacks, where the adversaries can implant malicious prediction behaviors into victim models by poisoning their training process. In this paper, we revisit poison-only backdoor attacks against speech recognition. We reveal that existing methods are not stealthy since their trigger patterns are perceptible to humans or machine detection. This limitation is mostly because their trigger patterns are simple noises or separable and distinctive clips. Motivated by these findings, we propose to exploit elements of sound ($e.g.$, pitch and timbre) to design more stealthy yet effective poison-only backdoor attacks. Specifically, we insert a short-duration high-pitched signal as the trigger and increase the pitch of remaining audio clips to `mask' it for designing stealthy pitch-based triggers. We manipulate timbre features of victim audios to design the stealthy timbre-based attack and design a voiceprint selection module to facilitate the multi-backdoor attack. Our attacks can generate more `natural' poisoned samples and therefore are more stealthy. Extensive experiments are conducted on benchmark datasets, which verify the effectiveness of our attacks under different settings ($e.g.$, all-to-one, all-to-all, clean-label, physical, and multi-backdoor settings) and their stealthiness. The code for reproducing main experiments are available at \url{https://github.com/HanboCai/BadSpeech_SoE}.



## **45. Certifying Model Accuracy under Distribution Shifts**

cs.LG

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2201.12440v3) [paper-pdf](http://arxiv.org/pdf/2201.12440v3)

**Authors**: Aounon Kumar, Alexander Levine, Tom Goldstein, Soheil Feizi

**Abstract**: Certified robustness in machine learning has primarily focused on adversarial perturbations of the input with a fixed attack budget for each point in the data distribution. In this work, we present provable robustness guarantees on the accuracy of a model under bounded Wasserstein shifts of the data distribution. We show that a simple procedure that randomizes the input of the model within a transformation space is provably robust to distributional shifts under the transformation. Our framework allows the datum-specific perturbation size to vary across different points in the input distribution and is general enough to include fixed-sized perturbations as well. Our certificates produce guaranteed lower bounds on the performance of the model for any (natural or adversarial) shift of the input distribution within a Wasserstein ball around the original distribution. We apply our technique to: (i) certify robustness against natural (non-adversarial) transformations of images such as color shifts, hue shifts and changes in brightness and saturation, (ii) certify robustness against adversarial shifts of the input distribution, and (iii) show provable lower bounds (hardness results) on the performance of models trained on so-called "unlearnable" datasets that have been poisoned to interfere with model training.



## **46. Training Socially Aligned Language Models in Simulated Human Society**

cs.CL

Code, data, and models can be downloaded via  https://github.com/agi-templar/Stable-Alignment

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2305.16960v2) [paper-pdf](http://arxiv.org/pdf/2305.16960v2)

**Authors**: Ruibo Liu, Ruixin Yang, Chenyan Jia, Ge Zhang, Denny Zhou, Andrew M. Dai, Diyi Yang, Soroush Vosoughi

**Abstract**: Social alignment in AI systems aims to ensure that these models behave according to established societal values. However, unlike humans, who derive consensus on value judgments through social interaction, current language models (LMs) are trained to rigidly replicate their training corpus in isolation, leading to subpar generalization in unfamiliar scenarios and vulnerability to adversarial attacks. This work presents a novel training paradigm that permits LMs to learn from simulated social interactions. In comparison to existing methodologies, our approach is considerably more scalable and efficient, demonstrating superior performance in alignment benchmarks and human evaluations. This paradigm shift in the training of LMs brings us a step closer to developing AI systems that can robustly and accurately reflect societal norms and values.



## **47. A First Order Meta Stackelberg Method for Robust Federated Learning**

cs.LG

Accepted to ICML 2023 Workshop on The 2nd New Frontiers In  Adversarial Machine Learning. Associated technical report arXiv:2306.13273

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2306.13800v3) [paper-pdf](http://arxiv.org/pdf/2306.13800v3)

**Authors**: Yunian Pan, Tao Li, Henger Li, Tianyi Xu, Zizhan Zheng, Quanyan Zhu

**Abstract**: Previous research has shown that federated learning (FL) systems are exposed to an array of security risks. Despite the proposal of several defensive strategies, they tend to be non-adaptive and specific to certain types of attacks, rendering them ineffective against unpredictable or adaptive threats. This work models adversarial federated learning as a Bayesian Stackelberg Markov game (BSMG) to capture the defender's incomplete information of various attack types. We propose meta-Stackelberg learning (meta-SL), a provably efficient meta-learning algorithm, to solve the equilibrium strategy in BSMG, leading to an adaptable FL defense. We demonstrate that meta-SL converges to the first-order $\varepsilon$-equilibrium point in $O(\varepsilon^{-2})$ gradient iterations, with $O(\varepsilon^{-4})$ samples needed per iteration, matching the state of the art. Empirical evidence indicates that our meta-Stackelberg framework performs exceptionally well against potent model poisoning and backdoor attacks of an uncertain nature.



## **48. Diffusion to Confusion: Naturalistic Adversarial Patch Generation Based on Diffusion Model for Object Detector**

cs.CV

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2307.08076v1) [paper-pdf](http://arxiv.org/pdf/2307.08076v1)

**Authors**: Shuo-Yen Lin, Ernie Chu, Che-Hsien Lin, Jun-Cheng Chen, Jia-Ching Wang

**Abstract**: Many physical adversarial patch generation methods are widely proposed to protect personal privacy from malicious monitoring using object detectors. However, they usually fail to generate satisfactory patch images in terms of both stealthiness and attack performance without making huge efforts on careful hyperparameter tuning. To address this issue, we propose a novel naturalistic adversarial patch generation method based on the diffusion models (DM). Through sampling the optimal image from the DM model pretrained upon natural images, it allows us to stably craft high-quality and naturalistic physical adversarial patches to humans without suffering from serious mode collapse problems as other deep generative models. To the best of our knowledge, we are the first to propose DM-based naturalistic adversarial patch generation for object detectors. With extensive quantitative, qualitative, and subjective experiments, the results demonstrate the effectiveness of the proposed approach to generate better-quality and more naturalistic adversarial patches while achieving acceptable attack performance than other state-of-the-art patch generation methods. We also show various generation trade-offs under different conditions.



## **49. How to choose your best allies for a transferable attack?**

cs.CR

ICCV 2023

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2304.02312v2) [paper-pdf](http://arxiv.org/pdf/2304.02312v2)

**Authors**: Thibault Maho, Seyed-Mohsen Moosavi-Dezfooli, Teddy Furon

**Abstract**: The transferability of adversarial examples is a key issue in the security of deep neural networks. The possibility of an adversarial example crafted for a source model fooling another targeted model makes the threat of adversarial attacks more realistic. Measuring transferability is a crucial problem, but the Attack Success Rate alone does not provide a sound evaluation. This paper proposes a new methodology for evaluating transferability by putting distortion in a central position. This new tool shows that transferable attacks may perform far worse than a black box attack if the attacker randomly picks the source model. To address this issue, we propose a new selection mechanism, called FiT, which aims at choosing the best source model with only a few preliminary queries to the target. Our experimental results show that FiT is highly effective at selecting the best source model for multiple scenarios such as single-model attacks, ensemble-model attacks and multiple attacks (Code available at: https://github.com/t-maho/transferability_measure_fit).



## **50. Byzantine-Robust Distributed Online Learning: Taming Adversarial Participants in An Adversarial Environment**

cs.LG

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2307.07980v1) [paper-pdf](http://arxiv.org/pdf/2307.07980v1)

**Authors**: Xingrong Dong, Zhaoxian Wu, Qing Ling, Zhi Tian

**Abstract**: This paper studies distributed online learning under Byzantine attacks. The performance of an online learning algorithm is often characterized by (adversarial) regret, which evaluates the quality of one-step-ahead decision-making when an environment provides adversarial losses, and a sublinear bound is preferred. But we prove that, even with a class of state-of-the-art robust aggregation rules, in an adversarial environment and in the presence of Byzantine participants, distributed online gradient descent can only achieve a linear adversarial regret bound, which is tight. This is the inevitable consequence of Byzantine attacks, even though we can control the constant of the linear adversarial regret to a reasonable level. Interestingly, when the environment is not fully adversarial so that the losses of the honest participants are i.i.d. (independent and identically distributed), we show that sublinear stochastic regret, in contrast to the aforementioned adversarial regret, is possible. We develop a Byzantine-robust distributed online momentum algorithm to attain such a sublinear stochastic regret bound. Extensive numerical experiments corroborate our theoretical analysis.



