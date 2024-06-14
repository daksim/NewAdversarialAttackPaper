# Latest Adversarial Attack Papers
**update at 2024-06-14 16:20:43**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Towards Evaluating the Robustness of Visual State Space Models**

cs.CV

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09407v1) [paper-pdf](http://arxiv.org/pdf/2406.09407v1)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Vision State Space Models (VSSMs), a novel architecture that combines the strengths of recurrent neural networks and latent variable models, have demonstrated remarkable performance in visual perception tasks by efficiently capturing long-range dependencies and modeling complex visual dynamics. However, their robustness under natural and adversarial perturbations remains a critical concern. In this work, we present a comprehensive evaluation of VSSMs' robustness under various perturbation scenarios, including occlusions, image structure, common corruptions, and adversarial attacks, and compare their performance to well-established architectures such as transformers and Convolutional Neural Networks. Furthermore, we investigate the resilience of VSSMs to object-background compositional changes on sophisticated benchmarks designed to test model performance in complex visual scenes. We also assess their robustness on object detection and segmentation tasks using corrupted datasets that mimic real-world scenarios. To gain a deeper understanding of VSSMs' adversarial robustness, we conduct a frequency analysis of adversarial attacks, evaluating their performance against low-frequency and high-frequency perturbations. Our findings highlight the strengths and limitations of VSSMs in handling complex visual corruptions, offering valuable insights for future research and improvements in this promising field. Our code and models will be available at https://github.com/HashmatShadab/MambaRobustness.



## **2. MirrorCheck: Efficient Adversarial Defense for Vision-Language Models**

cs.CV

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09250v1) [paper-pdf](http://arxiv.org/pdf/2406.09250v1)

**Authors**: Samar Fares, Klea Ziu, Toluwani Aremu, Nikita Durasov, Martin Takáč, Pascal Fua, Karthik Nandakumar, Ivan Laptev

**Abstract**: Vision-Language Models (VLMs) are becoming increasingly vulnerable to adversarial attacks as various novel attack strategies are being proposed against these models. While existing defenses excel in unimodal contexts, they currently fall short in safeguarding VLMs against adversarial threats. To mitigate this vulnerability, we propose a novel, yet elegantly simple approach for detecting adversarial samples in VLMs. Our method leverages Text-to-Image (T2I) models to generate images based on captions produced by target VLMs. Subsequently, we calculate the similarities of the embeddings of both input and generated images in the feature space to identify adversarial samples. Empirical evaluations conducted on different datasets validate the efficacy of our approach, outperforming baseline methods adapted from image classification domains. Furthermore, we extend our methodology to classification tasks, showcasing its adaptability and model-agnostic nature. Theoretical analyses and empirical findings also show the resilience of our approach against adaptive attacks, positioning it as an excellent defense mechanism for real-world deployment against adversarial threats.



## **3. After the Breach: Incident Response within Enterprises**

cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.07559v2) [paper-pdf](http://arxiv.org/pdf/2406.07559v2)

**Authors**: Sumanth Rao

**Abstract**: Enterprises are constantly under attack from sophisticated adversaries. These adversaries use a variety of techniques to first gain access to the enterprise, then spread laterally inside its networks, establish persistence, and finally exfiltrate sensitive data, or hold it for ransom. While historically, enterprises have used different Incident Response systems that monitor hosts, servers, or network devices to detect and report threats, these systems often need many analysts to triage and respond to alerts. However, the immense quantity of alerts to sift through, combined with the potential risk of missing a valid threat makes the task of the analyst challenging. To ease this manual and laborious process, researchers have proposed a variety of systems that perform automated attack investigations. These systems collect data, track causally related events, and present the analyst with an interpretable summary of the attack. In this paper, we present a survey of systems that perform automated attack investigation, and compare them based on their designs, goals, and heuristics. We discuss the challenges faced by these systems, and present a comparison in terms of their effectiveness, practicality, and ability to address these challenges. We conclude by discussing the future of these systems, and the open problems in this area.



## **4. Potion: Towards Poison Unlearning**

cs.LG

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09173v1) [paper-pdf](http://arxiv.org/pdf/2406.09173v1)

**Authors**: Stefan Schoepf, Jack Foster, Alexandra Brintrup

**Abstract**: Adversarial attacks by malicious actors on machine learning systems, such as introducing poison triggers into training datasets, pose significant risks. The challenge in resolving such an attack arises in practice when only a subset of the poisoned data can be identified. This necessitates the development of methods to remove, i.e. unlearn, poison triggers from already trained models with only a subset of the poison data available. The requirements for this task significantly deviate from privacy-focused unlearning where all of the data to be forgotten by the model is known. Previous work has shown that the undiscovered poisoned samples lead to a failure of established unlearning methods, with only one method, Selective Synaptic Dampening (SSD), showing limited success. Even full retraining, after the removal of the identified poison, cannot address this challenge as the undiscovered poison samples lead to a reintroduction of the poison trigger in the model. Our work addresses two key challenges to advance the state of the art in poison unlearning. First, we introduce a novel outlier-resistant method, based on SSD, that significantly improves model protection and unlearning performance. Second, we introduce Poison Trigger Neutralisation (PTN) search, a fast, parallelisable, hyperparameter search that utilises the characteristic "unlearning versus model protection" trade-off to find suitable hyperparameters in settings where the forget set size is unknown and the retain set is contaminated. We benchmark our contributions using ResNet-9 on CIFAR10 and WideResNet-28x10 on CIFAR100. Experimental results show that our method heals 93.72% of poison compared to SSD with 83.41% and full retraining with 40.68%. We achieve this while also lowering the average model accuracy drop caused by unlearning from 5.68% (SSD) to 1.41% (ours).



## **5. Beyond Labeling Oracles: What does it mean to steal ML models?**

cs.LG

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2310.01959v3) [paper-pdf](http://arxiv.org/pdf/2310.01959v3)

**Authors**: Avital Shafran, Ilia Shumailov, Murat A. Erdogdu, Nicolas Papernot

**Abstract**: Model extraction attacks are designed to steal trained models with only query access, as is often provided through APIs that ML-as-a-Service providers offer. Machine Learning (ML) models are expensive to train, in part because data is hard to obtain, and a primary incentive for model extraction is to acquire a model while incurring less cost than training from scratch. Literature on model extraction commonly claims or presumes that the attacker is able to save on both data acquisition and labeling costs. We thoroughly evaluate this assumption and find that the attacker often does not. This is because current attacks implicitly rely on the adversary being able to sample from the victim model's data distribution. We thoroughly research factors influencing the success of model extraction. We discover that prior knowledge of the attacker, i.e., access to in-distribution data, dominates other factors like the attack policy the adversary follows to choose which queries to make to the victim model API. Our findings urge the community to redefine the adversarial goals of ME attacks as current evaluation methods misinterpret the ME performance.



## **6. Hyper-parameter Tuning for Adversarially Robust Models**

cs.LG

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2304.02497v3) [paper-pdf](http://arxiv.org/pdf/2304.02497v3)

**Authors**: Pedro Mendes, Paolo Romano, David Garlan

**Abstract**: This work focuses on the problem of hyper-parameter tuning (HPT) for robust (i.e., adversarially trained) models, shedding light on the new challenges and opportunities arising during the HPT process for robust models. To this end, we conduct an extensive experimental study based on 3 popular deep models, in which we explore exhaustively 9 (discretized) HPs, 2 fidelity dimensions, and 2 attack bounds, for a total of 19208 configurations (corresponding to 50 thousand GPU hours). Through this study, we show that the complexity of the HPT problem is further exacerbated in adversarial settings due to the need to independently tune the HPs used during standard and adversarial training: succeeding in doing so (i.e., adopting different HP settings in both phases) can lead to a reduction of up to 80% and 43% of the error for clean and adversarial inputs, respectively. On the other hand, we also identify new opportunities to reduce the cost of HPT for robust models. Specifically, we propose to leverage cheap adversarial training methods to obtain inexpensive, yet highly correlated, estimations of the quality achievable using state-of-the-art methods. We show that, by exploiting this novel idea in conjunction with a recent multi-fidelity optimizer (taKG), the efficiency of the HPT process can be enhanced by up to 2.1x.



## **7. Improving Adversarial Robustness via Feature Pattern Consistency Constraint**

cs.CV

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.08829v1) [paper-pdf](http://arxiv.org/pdf/2406.08829v1)

**Authors**: Jiacong Hu, Jingwen Ye, Zunlei Feng, Jiazhen Yang, Shunyu Liu, Xiaotian Yu, Lingxiang Jia, Mingli Song

**Abstract**: Convolutional Neural Networks (CNNs) are well-known for their vulnerability to adversarial attacks, posing significant security concerns. In response to these threats, various defense methods have emerged to bolster the model's robustness. However, most existing methods either focus on learning from adversarial perturbations, leading to overfitting to the adversarial examples, or aim to eliminate such perturbations during inference, inevitably increasing computational burdens. Conversely, clean training, which strengthens the model's robustness by relying solely on clean examples, can address the aforementioned issues. In this paper, we align with this methodological stream and enhance its generalizability to unknown adversarial examples. This enhancement is achieved by scrutinizing the behavior of latent features within the network. Recognizing that a correct prediction relies on the correctness of the latent feature's pattern, we introduce a novel and effective Feature Pattern Consistency Constraint (FPCC) method to reinforce the latent feature's capacity to maintain the correct feature pattern. Specifically, we propose Spatial-wise Feature Modification and Channel-wise Feature Selection to enhance latent features. Subsequently, we employ the Pattern Consistency Loss to constrain the similarity between the feature pattern of the latent features and the correct feature pattern. Our experiments demonstrate that the FPCC method empowers latent features to uphold correct feature patterns even in the face of adversarial examples, resulting in inherent adversarial robustness surpassing state-of-the-art models.



## **8. Ranking Manipulation for Conversational Search Engines**

cs.CL

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.03589v2) [paper-pdf](http://arxiv.org/pdf/2406.03589v2)

**Authors**: Samuel Pfrommer, Yatong Bai, Tanmay Gautam, Somayeh Sojoudi

**Abstract**: Major search engine providers are rapidly incorporating Large Language Model (LLM)-generated content in response to user queries. These conversational search engines operate by loading retrieved website text into the LLM context for summarization and interpretation. Recent research demonstrates that LLMs are highly vulnerable to jailbreaking and prompt injection attacks, which disrupt the safety and quality goals of LLMs using adversarial strings. This work investigates the impact of prompt injections on the ranking order of sources referenced by conversational search engines. To this end, we introduce a focused dataset of real-world consumer product websites and formalize conversational search ranking as an adversarial problem. Experimentally, we analyze conversational search rankings in the absence of adversarial injections and show that different LLMs vary significantly in prioritizing product name, document content, and context position. We then present a tree-of-attacks-based jailbreaking technique which reliably promotes low-ranked products. Importantly, these attacks transfer effectively to state-of-the-art conversational search engines such as perplexity.ai. Given the strong financial incentive for website owners to boost their search ranking, we argue that our problem formulation is of critical importance for future robustness work.



## **9. On Security Weaknesses and Vulnerabilities in Deep Learning Systems**

cs.SE

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08688v1) [paper-pdf](http://arxiv.org/pdf/2406.08688v1)

**Authors**: Zhongzheng Lai, Huaming Chen, Ruoxi Sun, Yu Zhang, Minhui Xue, Dong Yuan

**Abstract**: The security guarantee of AI-enabled software systems (particularly using deep learning techniques as a functional core) is pivotal against the adversarial attacks exploiting software vulnerabilities. However, little attention has been paid to a systematic investigation of vulnerabilities in such systems. A common situation learned from the open source software community is that deep learning engineers frequently integrate off-the-shelf or open-source learning frameworks into their ecosystems. In this work, we specifically look into deep learning (DL) framework and perform the first systematic study of vulnerabilities in DL systems through a comprehensive analysis of identified vulnerabilities from Common Vulnerabilities and Exposures (CVE) and open-source DL tools, including TensorFlow, Caffe, OpenCV, Keras, and PyTorch. We propose a two-stream data analysis framework to explore vulnerability patterns from various databases. We investigate the unique DL frameworks and libraries development ecosystems that appear to be decentralized and fragmented. By revisiting the Common Weakness Enumeration (CWE) List, which provides the traditional software vulnerability related practices, we observed that it is more challenging to detect and fix the vulnerabilities throughout the DL systems lifecycle. Moreover, we conducted a large-scale empirical study of 3,049 DL vulnerabilities to better understand the patterns of vulnerability and the challenges in fixing them. We have released the full replication package at https://github.com/codelzz/Vulnerabilities4DLSystem. We anticipate that our study can advance the development of secure DL systems.



## **10. On Evaluating Adversarial Robustness of Volumetric Medical Segmentation Models**

eess.IV

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08486v1) [paper-pdf](http://arxiv.org/pdf/2406.08486v1)

**Authors**: Hashmat Shadab Malik, Numan Saeed, Asif Hanif, Muzammal Naseer, Mohammad Yaqub, Salman Khan, Fahad Shahbaz Khan

**Abstract**: Volumetric medical segmentation models have achieved significant success on organ and tumor-based segmentation tasks in recent years. However, their vulnerability to adversarial attacks remains largely unexplored, raising serious concerns regarding the real-world deployment of tools employing such models in the healthcare sector. This underscores the importance of investigating the robustness of existing models. In this context, our work aims to empirically examine the adversarial robustness across current volumetric segmentation architectures, encompassing Convolutional, Transformer, and Mamba-based models. We extend this investigation across four volumetric segmentation datasets, evaluating robustness under both white box and black box adversarial attacks. Overall, we observe that while both pixel and frequency-based attacks perform reasonably well under white box setting, the latter performs significantly better under transfer-based black box attacks. Across our experiments, we observe transformer-based models show higher robustness than convolution-based models with Mamba-based models being the most vulnerable. Additionally, we show that large-scale training of volumetric segmentation models improves the model's robustness against adversarial attacks. The code and pretrained models will be made available at https://github.com/HashmatShadab/Robustness-of-Volumetric-Medical-Segmentation-Models.



## **11. Transformation-Dependent Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08443v1) [paper-pdf](http://arxiv.org/pdf/2406.08443v1)

**Authors**: Yaoteng Tan, Zikui Cai, M. Salman Asif

**Abstract**: We introduce transformation-dependent adversarial attacks, a new class of threats where a single additive perturbation can trigger diverse, controllable mis-predictions by systematically transforming the input (e.g., scaling, blurring, compression). Unlike traditional attacks with static effects, our perturbations embed metamorphic properties to enable different adversarial attacks as a function of the transformation parameters. We demonstrate the transformation-dependent vulnerability across models (e.g., convolutional networks and vision transformers) and vision tasks (e.g., image classification and object detection). Our proposed geometric and photometric transformations enable a range of targeted errors from one crafted input (e.g., higher than 90% attack success rate for classifiers). We analyze effects of model architecture and type/variety of transformations on attack effectiveness. This work forces a paradigm shift by redefining adversarial inputs as dynamic, controllable threats. We highlight the need for robust defenses against such multifaceted, chameleon-like perturbations that current techniques are ill-prepared for.



## **12. Improving Noise Robustness through Abstractions and its Impact on Machine Learning**

cs.LG

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08428v1) [paper-pdf](http://arxiv.org/pdf/2406.08428v1)

**Authors**: Alfredo Ibias, Karol Capala, Varun Ravi Varma, Anna Drozdz, Jose Sousa

**Abstract**: Noise is a fundamental problem in learning theory with huge effects in the application of Machine Learning (ML) methods, due to real world data tendency to be noisy. Additionally, introduction of malicious noise can make ML methods fail critically, as is the case with adversarial attacks. Thus, finding and developing alternatives to improve robustness to noise is a fundamental problem in ML. In this paper, we propose a method to deal with noise: mitigating its effect through the use of data abstractions. The goal is to reduce the effect of noise over the model's performance through the loss of information produced by the abstraction. However, this information loss comes with a cost: it can result in an accuracy reduction due to the missing information. First, we explored multiple methodologies to create abstractions, using the training dataset, for the specific case of numerical data and binary classification tasks. We also tested how these abstractions can affect robustness to noise with several experiments that explore the robustness of an Artificial Neural Network to noise when trained using raw data \emph{vs} when trained using abstracted data. The results clearly show that using abstractions is a viable approach for developing noise robust ML methods.



## **13. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08298v1) [paper-pdf](http://arxiv.org/pdf/2406.08298v1)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.



## **14. Erasing Radio Frequency Fingerprints via Active Adversarial Perturbation**

cs.CR

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.07349v2) [paper-pdf](http://arxiv.org/pdf/2406.07349v2)

**Authors**: Zhaoyi Lu, Wenchao Xu, Ming Tu, Xin Xie, Cunqing Hua, Nan Cheng

**Abstract**: Radio Frequency (RF) fingerprinting is to identify a wireless device from its uniqueness of the analog circuitry or hardware imperfections. However, unlike the MAC address which can be modified, such hardware feature is inevitable for the signal emitted to air, which can possibly reveal device whereabouts, e.g., a sniffer can use a pre-trained model to identify a nearby device when receiving its signal. Such fingerprint may expose critical private information, e.g., the associated upper-layer applications or the end-user. In this paper, we propose to erase such RF feature for wireless devices, which can prevent fingerprinting by actively perturbation from the signal perspective. Specifically, we consider a common RF fingerprinting scenario, where machine learning models are trained from pilot signal data for identification. A novel adversarial attack solution is designed to generate proper perturbations, whereby the perturbed pilot signal can hide the hardware feature and misclassify the model. We theoretically show that the perturbation would not affect the communication function within a tolerable perturbation threshold. We also implement the pilot signal fingerprinting and the proposed perturbation process in a practical LTE system. Extensive experiment results demonstrate that the RF fingerprints can be effectively erased to protect the user privacy.



## **15. Adversarial Patch for 3D Local Feature Extractor**

cs.CV

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08102v1) [paper-pdf](http://arxiv.org/pdf/2406.08102v1)

**Authors**: Yu Wen Pao, Li Chang Lai, Hong-Yi Lin

**Abstract**: Local feature extractors are the cornerstone of many computer vision tasks. However, their vulnerability to adversarial attacks can significantly compromise their effectiveness. This paper discusses approaches to attack sophisticated local feature extraction algorithms and models to achieve two distinct goals: (1) forcing a match between originally non-matching image regions, and (2) preventing a match between originally matching regions. At the end of the paper, we discuss the performance and drawbacks of different patch generation methods.



## **16. Adversarial Evasion Attack Efficiency against Large Language Models**

cs.CL

9 pages, 1 table, 2 figures, DCAI 2024 conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08050v1) [paper-pdf](http://arxiv.org/pdf/2406.08050v1)

**Authors**: João Vitorino, Eva Maia, Isabel Praça

**Abstract**: Large Language Models (LLMs) are valuable for text classification, but their vulnerabilities must not be disregarded. They lack robustness against adversarial examples, so it is pertinent to understand the impacts of different types of perturbations, and assess if those attacks could be replicated by common users with a small amount of perturbations and a small number of queries to a deployed LLM. This work presents an analysis of the effectiveness, efficiency, and practicality of three different types of adversarial attacks against five different LLMs in a sentiment classification task. The obtained results demonstrated the very distinct impacts of the word-level and character-level attacks. The word attacks were more effective, but the character and more constrained attacks were more practical and required a reduced number of perturbations and queries. These differences need to be considered during the development of adversarial defense strategies to train more robust LLMs for intelligent text classification applications.



## **17. ADBA:Approximation Decision Boundary Approach for Black-Box Adversarial Attacks**

cs.LG

10 pages, 5 figures, conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.04998v2) [paper-pdf](http://arxiv.org/pdf/2406.04998v2)

**Authors**: Feiyang Wang, Xingquan Zuo, Hai Huang, Gang Chen

**Abstract**: Many machine learning models are susceptible to adversarial attacks, with decision-based black-box attacks representing the most critical threat in real-world applications. These attacks are extremely stealthy, generating adversarial examples using hard labels obtained from the target machine learning model. This is typically realized by optimizing perturbation directions, guided by decision boundaries identified through query-intensive exact search, significantly limiting the attack success rate. This paper introduces a novel approach using the Approximation Decision Boundary (ADB) to efficiently and accurately compare perturbation directions without precisely determining decision boundaries. The effectiveness of our ADB approach (ADBA) hinges on promptly identifying suitable ADB, ensuring reliable differentiation of all perturbation directions. For this purpose, we analyze the probability distribution of decision boundaries, confirming that using the distribution's median value as ADB can effectively distinguish different perturbation directions, giving rise to the development of the ADBA-md algorithm. ADBA-md only requires four queries on average to differentiate any pair of perturbation directions, which is highly query-efficient. Extensive experiments on six well-known image classifiers clearly demonstrate the superiority of ADBA and ADBA-md over multiple state-of-the-art black-box attacks. The source code is available at https://github.com/BUPTAIOC/ADBA.



## **18. Towards Reliable Empirical Machine Unlearning Evaluation: A Game-Theoretic View**

cs.LG

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2404.11577v2) [paper-pdf](http://arxiv.org/pdf/2404.11577v2)

**Authors**: Yiwen Tu, Pingbang Hu, Jiaqi Ma

**Abstract**: Machine unlearning is the process of updating machine learning models to remove the information of specific training data samples, in order to comply with data protection regulations that allow individuals to request the removal of their personal data. Despite the recent development of numerous unlearning algorithms, reliable evaluation of these algorithms remains an open research question. In this work, we focus on membership inference attack (MIA) based evaluation, one of the most common approaches for evaluating unlearning algorithms, and address various pitfalls of existing evaluation metrics that lack reliability. Specifically, we propose a game-theoretic framework that formalizes the evaluation process as a game between unlearning algorithms and MIA adversaries, measuring the data removal efficacy of unlearning algorithms by the capability of the MIA adversaries. Through careful design of the game, we demonstrate that the natural evaluation metric induced from the game enjoys provable guarantees that the existing evaluation metrics fail to satisfy. Furthermore, we propose a practical and efficient algorithm to estimate the evaluation metric induced from the game, and demonstrate its effectiveness through both theoretical analysis and empirical experiments. This work presents a novel and reliable approach to empirically evaluating unlearning algorithms, paving the way for the development of more effective unlearning techniques.



## **19. Graph Transductive Defense: a Two-Stage Defense for Graph Membership Inference Attacks**

cs.LG

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.07917v1) [paper-pdf](http://arxiv.org/pdf/2406.07917v1)

**Authors**: Peizhi Niu, Chao Pan, Siheng Chen, Olgica Milenkovic

**Abstract**: Graph neural networks (GNNs) have become instrumental in diverse real-world applications, offering powerful graph learning capabilities for tasks such as social networks and medical data analysis. Despite their successes, GNNs are vulnerable to adversarial attacks, including membership inference attacks (MIA), which threaten privacy by identifying whether a record was part of the model's training data. While existing research has explored MIA in GNNs under graph inductive learning settings, the more common and challenging graph transductive learning setting remains understudied in this context. This paper addresses this gap and proposes an effective two-stage defense, Graph Transductive Defense (GTD), tailored to graph transductive learning characteristics. The gist of our approach is a combination of a train-test alternate training schedule and flattening strategy, which successfully reduces the difference between the training and testing loss distributions. Extensive empirical results demonstrate the superior performance of our method (a decrease in attack AUROC by $9.42\%$ and an increase in utility performance by $18.08\%$ on average compared to LBP), highlighting its potential for seamless integration into various classification models with minimal overhead.



## **20. Mitigation of Channel Tampering Attacks in Continuous-Variable Quantum Key Distribution**

quant-ph

10 pages, 7 figures, closest to accepted version

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2401.15898v2) [paper-pdf](http://arxiv.org/pdf/2401.15898v2)

**Authors**: Sebastian P. Kish, Chandra Thapa, Mikhael Sayat, Hajime Suzuki, Josef Pieprzyk, Seyit Camtepe

**Abstract**: Despite significant advancements in continuous-variable quantum key distribution (CV-QKD), practical CV-QKD systems can be compromised by various attacks. Consequently, identifying new attack vectors and countermeasures for CV-QKD implementations is important for the continued robustness of CV-QKD. In particular, as CV-QKD relies on a public quantum channel, vulnerability to communication disruption persists from potential adversaries employing Denial-of-Service (DoS) attacks. Inspired by DoS attacks, this paper introduces a novel threat in CV-QKD called the Channel Amplification (CA) attack, wherein Eve manipulates the communication channel through amplification. We specifically model this attack in a CV-QKD optical fiber setup. To counter this threat, we propose a detection and mitigation strategy. Detection involves a machine learning (ML) model based on a decision tree classifier, classifying various channel tampering attacks, including CA and DoS attacks. For mitigation, Bob, post-selects quadrature data by classifying the attack type and frequency. Our ML model exhibits high accuracy in distinguishing and categorizing these attacks. The CA attack's impact on the secret key rate (SKR) is explored concerning Eve's location and the relative intensity noise of the local oscillator (LO). The proposed mitigation strategy improves the attacked SKR for CA attacks and, in some cases, for hybrid CA-DoS attacks. Our study marks a novel application of both ML classification and post-selection in this context. These findings are important for enhancing the robustness of CV-QKD systems against emerging threats on the channel.



## **21. Et Tu Certifications: Robustness Certificates Yield Better Adversarial Examples**

cs.LG

17 pages, 8 figures

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2302.04379v4) [paper-pdf](http://arxiv.org/pdf/2302.04379v4)

**Authors**: Andrew C. Cullen, Shijie Liu, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In guaranteeing the absence of adversarial examples in an instance's neighbourhood, certification mechanisms play an important role in demonstrating neural net robustness. In this paper, we ask if these certifications can compromise the very models they help to protect? Our new \emph{Certification Aware Attack} exploits certifications to produce computationally efficient norm-minimising adversarial examples $74 \%$ more often than comparable attacks, while reducing the median perturbation norm by more than $10\%$. While these attacks can be used to assess the tightness of certification bounds, they also highlight that releasing certifications can paradoxically reduce security.



## **22. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

cs.CL

19 Pages, 5 Figures, 8 Tables. Accepted by ACL 2024

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2309.14348v3) [paper-pdf](http://arxiv.org/pdf/2309.14348v3)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.



## **23. Adversarial Machine Unlearning**

cs.LG

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07687v1) [paper-pdf](http://arxiv.org/pdf/2406.07687v1)

**Authors**: Zonglin Di, Sixie Yu, Yevgeniy Vorobeychik, Yang Liu

**Abstract**: This paper focuses on the challenge of machine unlearning, aiming to remove the influence of specific training data on machine learning models. Traditionally, the development of unlearning algorithms runs parallel with that of membership inference attacks (MIA), a type of privacy threat to determine whether a data instance was used for training. However, the two strands are intimately connected: one can view machine unlearning through the lens of MIA success with respect to removed data. Recognizing this connection, we propose a game-theoretic framework that integrates MIAs into the design of unlearning algorithms. Specifically, we model the unlearning problem as a Stackelberg game in which an unlearner strives to unlearn specific training data from a model, while an auditor employs MIAs to detect the traces of the ostensibly removed data. Adopting this adversarial perspective allows the utilization of new attack advancements, facilitating the design of unlearning algorithms. Our framework stands out in two ways. First, it takes an adversarial approach and proactively incorporates the attacks into the design of unlearning algorithms. Secondly, it uses implicit differentiation to obtain the gradients that limit the attacker's success, thus benefiting the process of unlearning. We present empirical results to demonstrate the effectiveness of the proposed approach for machine unlearning.



## **24. Beware of Aliases -- Signal Preservation is Crucial for Robust Image Restoration**

cs.CV

Tags: Adversarial attack, image restoration, image deblurring,  frequency sampling

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07435v1) [paper-pdf](http://arxiv.org/pdf/2406.07435v1)

**Authors**: Shashank Agnihotri, Julia Grabinski, Janis Keuper, Margret Keuper

**Abstract**: Image restoration networks are usually comprised of an encoder and a decoder, responsible for aggregating image content from noisy, distorted data and to restore clean, undistorted images, respectively. Data aggregation as well as high-resolution image generation both usually come at the risk of involving aliases, i.e.~standard architectures put their ability to reconstruct the model input in jeopardy to reach high PSNR values on validation data. The price to be paid is low model robustness. In this work, we show that simply providing alias-free paths in state-of-the-art reconstruction transformers supports improved model robustness at low costs on the restoration performance. We do so by proposing BOA-Restormer, a transformer-based image restoration model that executes downsampling and upsampling operations partly in the frequency domain to ensure alias-free paths along the entire model while potentially preserving all relevant high-frequency information.



## **25. Text-CRS: A Generalized Certified Robustness Framework against Textual Adversarial Attacks**

cs.CR

Published in the 2024 IEEE Symposium on Security and Privacy (SP)

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2307.16630v2) [paper-pdf](http://arxiv.org/pdf/2307.16630v2)

**Authors**: Xinyu Zhang, Hanbin Hong, Yuan Hong, Peng Huang, Binghui Wang, Zhongjie Ba, Kui Ren

**Abstract**: The language models, especially the basic text classification models, have been shown to be susceptible to textual adversarial attacks such as synonym substitution and word insertion attacks. To defend against such attacks, a growing body of research has been devoted to improving the model robustness. However, providing provable robustness guarantees instead of empirical robustness is still widely unexplored. In this paper, we propose Text-CRS, a generalized certified robustness framework for natural language processing (NLP) based on randomized smoothing. To our best knowledge, existing certified schemes for NLP can only certify the robustness against $\ell_0$ perturbations in synonym substitution attacks. Representing each word-level adversarial operation (i.e., synonym substitution, word reordering, insertion, and deletion) as a combination of permutation and embedding transformation, we propose novel smoothing theorems to derive robustness bounds in both permutation and embedding space against such adversarial operations. To further improve certified accuracy and radius, we consider the numerical relationships between discrete words and select proper noise distributions for the randomized smoothing. Finally, we conduct substantial experiments on multiple language models and datasets. Text-CRS can address all four different word-level adversarial operations and achieve a significant accuracy improvement. We also provide the first benchmark on certified accuracy and radius of four word-level operations, besides outperforming the state-of-the-art certification against synonym substitution attacks.



## **26. Merging Improves Self-Critique Against Jailbreak Attacks**

cs.CL

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07188v1) [paper-pdf](http://arxiv.org/pdf/2406.07188v1)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .



## **27. Trainwreck: A damaging adversarial attack on image classifiers**

cs.CV

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2311.14772v2) [paper-pdf](http://arxiv.org/pdf/2311.14772v2)

**Authors**: Jan Zahálka

**Abstract**: Adversarial attacks are an important security concern for computer vision (CV). As CV models are becoming increasingly valuable assets in applied practice, disrupting them is emerging as a form of economic sabotage. This paper opens up the exploration of damaging adversarial attacks (DAAs) that seek to damage target CV models. DAAs are formalized by defining the threat model, the cost function DAAs maximize, and setting three requirements for success: potency, stealth, and customizability. As a pioneer DAA, this paper proposes Trainwreck, a train-time attack that conflates the data of similar classes in the training data using stealthy ($\epsilon \leq 8/255$) class-pair universal perturbations obtained from a surrogate model. Trainwreck is a black-box, transferable attack: it requires no knowledge of the target architecture, and a single poisoned dataset degrades the performance of any model trained on it. The experimental evaluation on CIFAR-10 and CIFAR-100 and various model architectures (EfficientNetV2, ResNeXt-101, and a finetuned ViT-L-16) demonstrates Trainwreck's efficiency. Trainwreck achieves similar or better potency compared to the data poisoning state of the art and is fully customizable by the poison rate parameter. Finally, data redundancy with hashing is identified as a reliable defense against Trainwreck or similar DAAs. The code is available at https://github.com/JanZahalka/trainwreck.



## **28. Benchmarking Trustworthiness of Multimodal Large Language Models: A Comprehensive Study**

cs.CL

100 pages, 84 figures, 33 tables

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07057v1) [paper-pdf](http://arxiv.org/pdf/2406.07057v1)

**Authors**: Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu

**Abstract**: Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust, the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/.



## **29. Adversarial flows: A gradient flow characterization of adversarial attacks**

cs.LG

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.05376v2) [paper-pdf](http://arxiv.org/pdf/2406.05376v2)

**Authors**: Lukas Weigand, Tim Roith, Martin Burger

**Abstract**: A popular method to perform adversarial attacks on neuronal networks is the so-called fast gradient sign method and its iterative variant. In this paper, we interpret this method as an explicit Euler discretization of a differential inclusion, where we also show convergence of the discretization to the associated gradient flow. To do so, we consider the concept of p-curves of maximal slope in the case $p=\infty$. We prove existence of $\infty$-curves of maximum slope and derive an alternative characterization via differential inclusions. Furthermore, we also consider Wasserstein gradient flows for potential energies, where we show that curves in the Wasserstein space can be characterized by a representing measure on the space of curves in the underlying Banach space, which fulfill the differential inclusion. The application of our theory to the finite-dimensional setting is twofold: On the one hand, we show that a whole class of normalized gradient descent methods (in particular signed gradient descent) converge, up to subsequences, to the flow, when sending the step size to zero. On the other hand, in the distributional setting, we show that the inner optimization task of adversarial training objective can be characterized via $\infty$-curves of maximum slope on an appropriate optimal transport space.



## **30. Post-train Black-box Defense via Bayesian Boundary Correction**

cs.CV

arXiv admin note: text overlap with arXiv:2203.04713

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2306.16979v3) [paper-pdf](http://arxiv.org/pdf/2306.16979v3)

**Authors**: He Wang, Yunfeng Diao

**Abstract**: Classifiers based on deep neural networks are susceptible to adversarial attack, where the widely existing vulnerability has invoked the research in defending them from potential threats. Given a vulnerable classifier, existing defense methods are mostly white-box and often require re-training the victim under modified loss functions/training regimes. While the model/data/training specifics of the victim are usually unavailable to the user, re-training is unappealing, if not impossible for reasons such as limited computational resources. To this end, we propose a new post-train black-box defense framework. It can turn any pre-trained classifier into a resilient one with little knowledge of the model specifics. This is achieved by new joint Bayesian treatments on the clean data, the adversarial examples and the classifier, for maximizing their joint probability. It is further equipped with a new post-train strategy which keeps the victim intact, avoiding re-training. We name our framework Bayesian Boundary Correction (BBC). BBC is a general and flexible framework that can easily adapt to different data types. We instantiate BBC for image classification and skeleton-based human activity recognition, for both static and dynamic data. Exhaustive evaluation shows that BBC has superior robustness and can enhance robustness without severely hurting the clean accuracy, compared with existing defense methods.



## **31. DISCO Might Not Be Funky: Random Intelligent Reflective Surface Configurations That Attack**

eess.SP

This paper has been accepted by IEEE Wireless Communications. For the  code of the DISCO RIS is available on Github  (https://github.com/huanhuan1799/Disco-Intelligent-Reflecting-Surfaces-Active-Channel-Aging-for-Fully-Passive-Jamming-Attacks)

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2310.00687v2) [paper-pdf](http://arxiv.org/pdf/2310.00687v2)

**Authors**: Huan Huang, Lipeng Dai, Hongliang Zhang, Chongfu Zhang, Zhongxing Tian, Yi Cai, A. Lee Swindlehurst, Zhu Han

**Abstract**: Emerging intelligent reflective surfaces (IRSs) significantly improve system performance, but also pose a significant risk for physical layer security (PLS). Unlike the extensive research on legitimate IRS-enhanced communications, in this article we present an adversarial IRS-based fully-passive jammer (FPJ). We describe typical application scenarios for Disco IRS (DIRS)-based FPJ, where an illegitimate IRS with random, time-varying reflection properties acts like a "disco ball" to randomly change the propagation environment. We introduce the principles of DIRS-based FPJ and overview existing investigations of the technology, including a design example employing one-bit phase shifters. The DIRS-based FPJ can be implemented without either jamming power or channel state information (CSI) for the legitimate users (LUs). It does not suffer from the energy constraints of traditional active jammers, nor does it require any knowledge of the LU channels. In addition to the proposed jamming attack, we also propose an anti-jamming strategy that requires only statistical rather than instantaneous CSI. Furthermore, we present a data frame structure that enables the legitimate access point (AP) to estimate the DIRS-jammed channels' statistical characteristics in the presence of the DIRS jamming. Typical cases are discussed to show the impact of the DIRS-based FPJ and the feasibility of the anti-jamming precoder (AJP). Moreover, we outline future research directions and challenges for the DIRS-based FPJ and its anti-jamming precoding to stimulate this line of research and pave the way for practical applications.



## **32. Reinforced Compressive Neural Architecture Search for Versatile Adversarial Robustness**

cs.LG

17 pages

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06792v1) [paper-pdf](http://arxiv.org/pdf/2406.06792v1)

**Authors**: Dingrong Wang, Hitesh Sapkota, Zhiqiang Tao, Qi Yu

**Abstract**: Prior neural architecture search (NAS) for adversarial robustness works have discovered that a lightweight and adversarially robust neural network architecture could exist in a non-robust large teacher network, generally disclosed by heuristic rules through statistical analysis and neural architecture search, generally disclosed by heuristic rules from neural architecture search. However, heuristic methods cannot uniformly handle different adversarial attacks and "teacher" network capacity. To solve this challenge, we propose a Reinforced Compressive Neural Architecture Search (RC-NAS) for Versatile Adversarial Robustness. Specifically, we define task settings that compose datasets, adversarial attacks, and teacher network information. Given diverse tasks, we conduct a novel dual-level training paradigm that consists of a meta-training and a fine-tuning phase to effectively expose the RL agent to diverse attack scenarios (in meta-training), and making it adapt quickly to locate a sub-network (in fine-tuning) for any previously unseen scenarios. Experiments show that our framework could achieve adaptive compression towards different initial teacher networks, datasets, and adversarial attacks, resulting in more lightweight and adversarially robust architectures.



## **33. Robust Distribution Learning with Local and Global Adversarial Corruptions**

cs.LG

Accepted for presentation at the Conference on Learning Theory (COLT)  2024

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06509v1) [paper-pdf](http://arxiv.org/pdf/2406.06509v1)

**Authors**: Sloan Nietert, Ziv Goldfeld, Soroosh Shafiee

**Abstract**: We consider learning in an adversarial environment, where an $\varepsilon$-fraction of samples from a distribution $P$ are arbitrarily modified (*global* corruptions) and the remaining perturbations have average magnitude bounded by $\rho$ (*local* corruptions). Given access to $n$ such corrupted samples, we seek a computationally efficient estimator $\hat{P}_n$ that minimizes the Wasserstein distance $\mathsf{W}_1(\hat{P}_n,P)$. In fact, we attack the fine-grained task of minimizing $\mathsf{W}_1(\Pi_\# \hat{P}_n, \Pi_\# P)$ for all orthogonal projections $\Pi \in \mathbb{R}^{d \times d}$, with performance scaling with $\mathrm{rank}(\Pi) = k$. This allows us to account simultaneously for mean estimation ($k=1$), distribution estimation ($k=d$), as well as the settings interpolating between these two extremes. We characterize the optimal population-limit risk for this task and then develop an efficient finite-sample algorithm with error bounded by $\sqrt{\varepsilon k} + \rho + d^{O(1)}\tilde{O}(n^{-1/k})$ when $P$ has bounded moments of order $2+\delta$, for constant $\delta > 0$. For data distributions with bounded covariance, our finite-sample bounds match the minimax population-level optimum for large sample sizes. Our efficient procedure relies on a novel trace norm approximation of an ideal yet intractable 2-Wasserstein projection estimator. We apply this algorithm to robust stochastic optimization, and, in the process, uncover a new method for overcoming the curse of dimensionality in Wasserstein distributionally robust optimization.



## **34. Improving Alignment and Robustness with Circuit Breakers**

cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.04313v2) [paper-pdf](http://arxiv.org/pdf/2406.04313v2)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.



## **35. Evolving Assembly Code in an Adversarial Environment**

cs.NE

20 pages, 6 figures, 6 listings, 5 tables

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2403.19489v2) [paper-pdf](http://arxiv.org/pdf/2403.19489v2)

**Authors**: Irina Maliukov, Gera Weiss, Oded Margalit, Achiya Elyasaf

**Abstract**: In this work, we evolve Assembly code for the CodeGuru competition. The goal is to create a survivor -- an Assembly program that runs the longest in shared memory, by resisting attacks from adversary survivors and finding their weaknesses. For evolving top-notch solvers, we specify a Backus Normal Form (BNF) for the Assembly language and synthesize the code from scratch using Genetic Programming (GP). We evaluate the survivors by running CodeGuru games against human-written winning survivors. Our evolved programs found weaknesses in the programs they were trained against and utilized them. To push evolution further, we implemented memetic operators that utilize machine learning to explore the solution space effectively. This work has important applications for cyber-security as we utilize evolution to detect weaknesses in survivors. The Assembly BNF is domain-independent; thus, by modifying the fitness function, it can detect code weaknesses and help fix them. Finally, the CodeGuru competition offers a novel platform for analyzing GP and code evolution in adversarial environments. To support further research in this direction, we provide a thorough qualitative analysis of the evolved survivors and the weaknesses found.



## **36. Explainable Graph Neural Networks Under Fire**

cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06417v1) [paper-pdf](http://arxiv.org/pdf/2406.06417v1)

**Authors**: Zhong Li, Simon Geisler, Yuhang Wang, Stephan Günnemann, Matthijs van Leeuwen

**Abstract**: Predictions made by graph neural networks (GNNs) usually lack interpretability due to their complex computational behavior and the abstract nature of graphs. In an attempt to tackle this, many GNN explanation methods have emerged. Their goal is to explain a model's predictions and thereby obtain trust when GNN models are deployed in decision critical applications. Most GNN explanation methods work in a post-hoc manner and provide explanations in the form of a small subset of important edges and/or nodes. In this paper we demonstrate that these explanations can unfortunately not be trusted, as common GNN explanation methods turn out to be highly susceptible to adversarial perturbations. That is, even small perturbations of the original graph structure that preserve the model's predictions may yield drastically different explanations. This calls into question the trustworthiness and practical utility of post-hoc explanation methods for GNNs. To be able to attack GNN explanation models, we devise a novel attack method dubbed \textit{GXAttack}, the first \textit{optimization-based} adversarial attack method for post-hoc GNN explanations under such settings. Due to the devastating effectiveness of our attack, we call for an adversarial evaluation of future GNN explainers to demonstrate their robustness.



## **37. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors**

cs.CL

ACL 2024

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2405.07940v2) [paper-pdf](http://arxiv.org/pdf/2405.07940v2)

**Authors**: Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, Chris Callison-Burch

**Abstract**: Many commercial and open-source models claim to detect machine-generated text with extremely high accuracy (99% or more). However, very few of these detectors are evaluated on shared benchmark datasets and even when they are, the datasets used for evaluation are insufficiently challenging-lacking variations in sampling strategy, adversarial attacks, and open-source generative models. In this work we present RAID: the largest and most challenging benchmark dataset for machine-generated text detection. RAID includes over 6 million generations spanning 11 models, 8 domains, 11 adversarial attacks and 4 decoding strategies. Using RAID, we evaluate the out-of-domain and adversarial robustness of 8 open- and 4 closed-source detectors and find that current detectors are easily fooled by adversarial attacks, variations in sampling strategies, repetition penalties, and unseen generative models. We release our data along with a leaderboard to encourage future research.



## **38. Towards Transferable Targeted 3D Adversarial Attack in the Physical World**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2312.09558v3) [paper-pdf](http://arxiv.org/pdf/2312.09558v3)

**Authors**: Yao Huang, Yinpeng Dong, Shouwei Ruan, Xiao Yang, Hang Su, Xingxing Wei

**Abstract**: Compared with transferable untargeted attacks, transferable targeted adversarial attacks could specify the misclassification categories of adversarial samples, posing a greater threat to security-critical tasks. In the meanwhile, 3D adversarial samples, due to their potential of multi-view robustness, can more comprehensively identify weaknesses in existing deep learning systems, possessing great application value. However, the field of transferable targeted 3D adversarial attacks remains vacant. The goal of this work is to develop a more effective technique that could generate transferable targeted 3D adversarial examples, filling the gap in this field. To achieve this goal, we design a novel framework named TT3D that could rapidly reconstruct from few multi-view images into Transferable Targeted 3D textured meshes. While existing mesh-based texture optimization methods compute gradients in the high-dimensional mesh space and easily fall into local optima, leading to unsatisfactory transferability and distinct distortions, TT3D innovatively performs dual optimization towards both feature grid and Multi-layer Perceptron (MLP) parameters in the grid-based NeRF space, which significantly enhances black-box transferability while enjoying naturalness. Experimental results show that TT3D not only exhibits superior cross-model transferability but also maintains considerable adaptability across different renders and vision tasks. More importantly, we produce 3D adversarial examples with 3D printing techniques in the real world and verify their robust performance under various scenarios.



## **39. Siren -- Advancing Cybersecurity through Deception and Adaptive Analysis**

cs.CR

7 pages, 6 figures

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06225v1) [paper-pdf](http://arxiv.org/pdf/2406.06225v1)

**Authors**: Girish Kulathumani, Samruth Ananthanarayanan, Ganesh Narayanan

**Abstract**: Siren represents a pioneering research effort aimed at fortifying cybersecurity through strategic integration of deception, machine learning, and proactive threat analysis. Drawing inspiration from mythical sirens, this project employs sophisticated methods to lure potential threats into controlled environments. The system features a dynamic machine learning model for real-time analysis and classification, ensuring continuous adaptability to emerging cyber threats. The architectural framework includes a link monitoring proxy, a purpose-built machine learning model for dynamic link analysis, and a honeypot enriched with simulated user interactions to intensify threat engagement. Data protection within the honeypot is fortified with probabilistic encryption. Additionally, the incorporation of simulated user activity extends the system's capacity to capture and learn from potential attackers even after user disengagement. Siren introduces a paradigm shift in cybersecurity, transforming traditional defense mechanisms into proactive systems that actively engage and learn from potential adversaries. The research strives to enhance user protection while yielding valuable insights for ongoing refinement in response to the evolving landscape of cybersecurity threats.



## **40. Defending Against Physical Adversarial Patch Attacks on Infrared Human Detection**

cs.CV

Accepted at ICIP2024. Lukas Strack and Futa Waseda contributed  equally

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2309.15519v3) [paper-pdf](http://arxiv.org/pdf/2309.15519v3)

**Authors**: Lukas Strack, Futa Waseda, Huy H. Nguyen, Yinqiang Zheng, Isao Echizen

**Abstract**: Infrared detection is an emerging technique for safety-critical tasks owing to its remarkable anti-interference capability. However, recent studies have revealed that it is vulnerable to physically-realizable adversarial patches, posing risks in its real-world applications. To address this problem, we are the first to investigate defense strategies against adversarial patch attacks on infrared detection, especially human detection. We propose a straightforward defense strategy, patch-based occlusion-aware detection (POD), which efficiently augments training samples with random patches and subsequently detects them. POD not only robustly detects people but also identifies adversarial patch locations. Surprisingly, while being extremely computationally efficient, POD easily generalizes to state-of-the-art adversarial patch attacks that are unseen during training. Furthermore, POD improves detection precision even in a clean (i.e., no-attack) situation due to the data augmentation effect. Our evaluation demonstrates that POD is robust to adversarial patches of various shapes and sizes. The effectiveness of our baseline approach is shown to be a viable defense mechanism for real-world infrared human detection systems, paving the way for exploring future research directions.



## **41. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2402.18104v2) [paper-pdf](http://arxiv.org/pdf/2402.18104v2)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and closed-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 91.1% attack success rate on OpenAI GPT-4 chatbot.



## **42. Texture Re-scalable Universal Adversarial Perturbation**

cs.CV

14 pages (accepted by TIFS2024)

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06089v1) [paper-pdf](http://arxiv.org/pdf/2406.06089v1)

**Authors**: Yihao Huang, Qing Guo, Felix Juefei-Xu, Ming Hu, Xiaojun Jia, Xiaochun Cao, Geguang Pu, Yang Liu

**Abstract**: Universal adversarial perturbation (UAP), also known as image-agnostic perturbation, is a fixed perturbation map that can fool the classifier with high probabilities on arbitrary images, making it more practical for attacking deep models in the real world. Previous UAP methods generate a scale-fixed and texture-fixed perturbation map for all images, which ignores the multi-scale objects in images and usually results in a low fooling ratio. Since the widely used convolution neural networks tend to classify objects according to semantic information stored in local textures, it seems a reasonable and intuitive way to improve the UAP from the perspective of utilizing local contents effectively. In this work, we find that the fooling ratios significantly increase when we add a constraint to encourage a small-scale UAP map and repeat it vertically and horizontally to fill the whole image domain. To this end, we propose texture scale-constrained UAP (TSC-UAP), a simple yet effective UAP enhancement method that automatically generates UAPs with category-specific local textures that can fool deep models more easily. Through a low-cost operation that restricts the texture scale, TSC-UAP achieves a considerable improvement in the fooling ratio and attack transferability for both data-dependent and data-free UAP methods. Experiments conducted on two state-of-the-art UAP methods, eight popular CNN models and four classical datasets show the remarkable performance of TSC-UAP.



## **43. When Authentication Is Not Enough: On the Security of Behavioral-Based Driver Authentication Systems**

cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2306.05923v4) [paper-pdf](http://arxiv.org/pdf/2306.05923v4)

**Authors**: Emad Efatinasab, Francesco Marchiori, Denis Donadel, Alessandro Brighente, Mauro Conti

**Abstract**: Many research papers have recently focused on behavioral-based driver authentication systems in vehicles. Pushed by Artificial Intelligence (AI) advancements, these works propose powerful models to identify drivers through their unique biometric behavior. However, these models have never been scrutinized from a security point of view, rather focusing on the performance of the AI algorithms. Several limitations and oversights make implementing the state-of-the-art impractical, such as their secure connection to the vehicle's network and the management of security alerts. Furthermore, due to the extensive use of AI, these systems may be vulnerable to adversarial attacks. However, there is currently no discussion on the feasibility and impact of such attacks in this scenario.   Driven by the significant gap between research and practical application, this paper seeks to connect these two domains. We propose the first security-aware system model for behavioral-based driver authentication. We develop two lightweight driver authentication systems based on Random Forest and Recurrent Neural Network architectures designed for our constrained environments. We formalize a realistic system and threat model reflecting a real-world vehicle's network for their implementation. When evaluated on real driving data, our models outclass the state-of-the-art with an accuracy of up to 0.999 in identification and authentication. Moreover, we are the first to propose attacks against these systems by developing two novel evasion attacks, SMARTCAN and GANCAN. We show how attackers can still exploit these systems with a perfect attack success rate (up to 1.000). Finally, we discuss requirements for deploying driver authentication systems securely. Through our contributions, we aid practitioners in safely adopting these systems, help reduce car thefts, and enhance driver security.



## **44. A High Dimensional Statistical Model for Adversarial Training: Geometry and Trade-Offs**

stat.ML

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2402.05674v2) [paper-pdf](http://arxiv.org/pdf/2402.05674v2)

**Authors**: Kasimir Tanner, Matteo Vilucchio, Bruno Loureiro, Florent Krzakala

**Abstract**: This work investigates adversarial training in the context of margin-based linear classifiers in the high-dimensional regime where the dimension $d$ and the number of data points $n$ diverge with a fixed ratio $\alpha = n / d$. We introduce a tractable mathematical model where the interplay between the data and adversarial attacker geometries can be studied, while capturing the core phenomenology observed in the adversarial robustness literature. Our main theoretical contribution is an exact asymptotic description of the sufficient statistics for the adversarial empirical risk minimiser, under generic convex and non-increasing losses. Our result allow us to precisely characterise which directions in the data are associated with a higher generalisation/robustness trade-off, as defined by a robustness and a usefulness metric. In particular, we unveil the existence of directions which can be defended without penalising accuracy. Finally, we show the advantage of defending non-robust features during training, identifying a uniform protection as an inherently effective defence mechanism.



## **45. Safety Alignment Should Be Made More Than Just a Few Tokens Deep**

cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.05946v1) [paper-pdf](http://arxiv.org/pdf/2406.05946v1)

**Authors**: Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, Peter Henderson

**Abstract**: The safety alignment of current Large Language Models (LLMs) is vulnerable. Relatively simple attacks, or even benign fine-tuning, can jailbreak aligned models. We argue that many of these vulnerabilities are related to a shared underlying issue: safety alignment can take shortcuts, wherein the alignment adapts a model's generative distribution primarily over only its very first few output tokens. We refer to this issue as shallow safety alignment. In this paper, we present case studies to explain why shallow safety alignment can exist and provide evidence that current aligned LLMs are subject to this issue. We also show how these findings help explain multiple recently discovered vulnerabilities in LLMs, including the susceptibility to adversarial suffix attacks, prefilling attacks, decoding parameter attacks, and fine-tuning attacks. Importantly, we discuss how this consolidated notion of shallow safety alignment sheds light on promising research directions for mitigating these vulnerabilities. For instance, we show that deepening the safety alignment beyond just the first few tokens can often meaningfully improve robustness against some common exploits. Finally, we design a regularized finetuning objective that makes the safety alignment more persistent against fine-tuning attacks by constraining updates on initial tokens. Overall, we advocate that future safety alignment should be made more than just a few tokens deep.



## **46. A Relevance Model for Threat-Centric Ranking of Cybersecurity Vulnerabilities**

cs.CR

24 pages, 8 figures, 14 tables

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05933v1) [paper-pdf](http://arxiv.org/pdf/2406.05933v1)

**Authors**: Corren McCoy, Ross Gore, Michael L. Nelson, Michele C. Weigle

**Abstract**: The relentless process of tracking and remediating vulnerabilities is a top concern for cybersecurity professionals. The key challenge is trying to identify a remediation scheme specific to in-house, organizational objectives. Without a strategy, the result is a patchwork of fixes applied to a tide of vulnerabilities, any one of which could be the point of failure in an otherwise formidable defense. Given that few vulnerabilities are a focus of real-world attacks, a practical remediation strategy is to identify vulnerabilities likely to be exploited and focus efforts towards remediating those vulnerabilities first. The goal of this research is to demonstrate that aggregating and synthesizing readily accessible, public data sources to provide personalized, automated recommendations for organizations to prioritize their vulnerability management strategy will offer significant improvements over using the Common Vulnerability Scoring System (CVSS). We provide a framework for vulnerability management specifically focused on mitigating threats using adversary criteria derived from MITRE ATT&CK. We test our approach by identifying vulnerabilities in software associated with six universities and four government facilities. Ranking policy performance is measured using the Normalized Discounted Cumulative Gain (nDCG). Our results show an average 71.5% - 91.3% improvement towards the identification of vulnerabilities likely to be targeted and exploited by cyber threat actors. The return on investment (ROI) of patching using our policies results in a savings of 23.3% - 25.5% in annualized costs. Our results demonstrate the efficacy of creating knowledge graphs to link large data sets to facilitate semantic queries and create data-driven, flexible ranking policies.



## **47. MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification**

cs.CV

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05927v1) [paper-pdf](http://arxiv.org/pdf/2406.05927v1)

**Authors**: Sajjad Amini, Mohammadreza Teymoorianfard, Shiqing Ma, Amir Houmansadr

**Abstract**: We present a simple yet effective method to improve the robustness of Convolutional Neural Networks (CNNs) against adversarial examples by post-processing an adversarially trained model. Our technique, MeanSparse, cascades the activation functions of a trained model with novel operators that sparsify mean-centered feature vectors. This is equivalent to reducing feature variations around the mean, and we show that such reduced variations merely affect the model's utility, yet they strongly attenuate the adversarial perturbations and decrease the attacker's success rate. Our experiments show that, when applied to the top models in the RobustBench leaderboard, it achieves a new robustness record of 72.08% (from 71.07%) and 59.64% (from 59.56%) on CIFAR-10 and ImageNet, respectively, in term of AutoAttack accuracy. Code is available at https://github.com/SPIN-UMass/MeanSparse



## **48. Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents**

cs.CR

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05870v1) [paper-pdf](http://arxiv.org/pdf/2406.05870v1)

**Authors**: Avital Shafran, Roei Schuster, Vitaly Shmatikov

**Abstract**: Retrieval-augmented generation (RAG) systems respond to queries by retrieving relevant documents from a knowledge database, then generating an answer by applying an LLM to the retrieved documents.   We demonstrate that RAG systems that operate on databases with potentially untrusted content are vulnerable to a new class of denial-of-service attacks we call jamming. An adversary can add a single ``blocker'' document to the database that will be retrieved in response to a specific query and, furthermore, result in the RAG system not answering the query - ostensibly because it lacks the information or because the answer is unsafe.   We describe and analyze several methods for generating blocker documents, including a new method based on black-box optimization that does not require the adversary to know the embedding or LLM used by the target RAG system, nor access to an auxiliary LLM to generate blocker documents. We measure the efficacy of the considered methods against several LLMs and embeddings, and demonstrate that the existing safety metrics for LLMs do not capture their vulnerability to jamming. We then discuss defenses against blocker documents.



## **49. Self-supervised Adversarial Training of Monocular Depth Estimation against Physical-World Attacks**

cs.CV

Accepted in TPAMI'24. Extended from our ICLR'23 publication  (arXiv:2301.13487). arXiv admin note: substantial text overlap with  arXiv:2301.13487

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05857v1) [paper-pdf](http://arxiv.org/pdf/2406.05857v1)

**Authors**: Zhiyuan Cheng, Cheng Han, James Liang, Qifan Wang, Xiangyu Zhang, Dongfang Liu

**Abstract**: Monocular Depth Estimation (MDE) plays a vital role in applications such as autonomous driving. However, various attacks target MDE models, with physical attacks posing significant threats to system security. Traditional adversarial training methods, which require ground-truth labels, are not directly applicable to MDE models that lack ground-truth depth. Some self-supervised model hardening techniques (e.g., contrastive learning) overlook the domain knowledge of MDE, resulting in suboptimal performance. In this work, we introduce a novel self-supervised adversarial training approach for MDE models, leveraging view synthesis without the need for ground-truth depth. We enhance adversarial robustness against real-world attacks by incorporating L_0-norm-bounded perturbation during training. We evaluate our method against supervised learning-based and contrastive learning-based approaches specifically designed for MDE. Our experiments with two representative MDE networks demonstrate improved robustness against various adversarial attacks, with minimal impact on benign performance.



## **50. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

cs.LG

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2402.06255v2) [paper-pdf](http://arxiv.org/pdf/2402.06255v2)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreak attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly with a particular focus on harmful content filtering or heuristical defensive prompt designs. However, how to achieve intrinsic robustness through the prompts remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both black-box and white-box attacks, reducing the success rate of advanced attacks to nearly 0 while maintaining the model's utility on the benign task. The proposed defense strategy incurs only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/rain152/PAT.



