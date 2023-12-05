# Latest Adversarial Attack Papers
**update at 2023-12-05 21:17:12**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01886v1) [paper-pdf](http://arxiv.org/pdf/2312.01886v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.



## **2. Two-stage optimized unified adversarial patch for attacking visible-infrared cross-modal detectors in the physical world**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01789v1) [paper-pdf](http://arxiv.org/pdf/2312.01789v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Currently, many studies have addressed security concerns related to visible and infrared detectors independently. In practical scenarios, utilizing cross-modal detectors for tasks proves more reliable than relying on single-modal detectors. Despite this, there is a lack of comprehensive security evaluations for cross-modal detectors. While existing research has explored the feasibility of attacks against cross-modal detectors, the implementation of a robust attack remains unaddressed. This work introduces the Two-stage Optimized Unified Adversarial Patch (TOUAP) designed for performing attacks against visible-infrared cross-modal detectors in real-world, black-box settings. The TOUAP employs a two-stage optimization process: firstly, PSO optimizes an irregular polygonal infrared patch to attack the infrared detector; secondly, the color QR code is optimized, and the shape information of the infrared patch from the first stage is used as a mask. The resulting irregular polygon visible modal patch executes an attack on the visible detector. Through extensive experiments conducted in both digital and physical environments, we validate the effectiveness and robustness of the proposed method. As the TOUAP surpasses baseline performance, we advocate for its widespread attention.



## **3. Malicious Lateral Movement in 5G Core With Network Slicing And Its Detection**

cs.CR

Accepted for publication in the Proceedings of IEEE ITNAC-2023

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01681v1) [paper-pdf](http://arxiv.org/pdf/2312.01681v1)

**Authors**: Ayush Kumar, Vrizlynn L. L. Thing

**Abstract**: 5G networks are susceptible to cyber attacks due to reasons such as implementation issues and vulnerabilities in 3GPP standard specifications. In this work, we propose lateral movement strategies in a 5G Core (5GC) with network slicing enabled, as part of a larger attack campaign by well-resourced adversaries such as APT groups. Further, we present 5GLatte, a system to detect such malicious lateral movement. 5GLatte operates on a host-container access graph built using host/NF container logs collected from the 5GC. Paths inferred from the access graph are scored based on selected filtering criteria and subsequently presented as input to a threshold-based anomaly detection algorithm to reveal malicious lateral movement paths. We evaluate 5GLatte on a dataset containing attack campaigns (based on MITRE ATT&CK and FiGHT frameworks) launched in a 5G test environment which shows that compared to other lateral movement detectors based on state-of-the-art, it can achieve higher true positive rates with similar false positive rates.



## **4. Adversarial Medical Image with Hierarchical Feature Hiding**

eess.IV

Our code is available at  \url{https://github.com/qsyao/Hierarchical_Feature_Constraint}

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01679v1) [paper-pdf](http://arxiv.org/pdf/2312.01679v1)

**Authors**: Qingsong Yao, Zecheng He, Yuexiang Li, Yi Lin, Kai Ma, Yefeng Zheng, S. Kevin Zhou

**Abstract**: Deep learning based methods for medical images can be easily compromised by adversarial examples (AEs), posing a great security flaw in clinical decision-making. It has been discovered that conventional adversarial attacks like PGD which optimize the classification logits, are easy to distinguish in the feature space, resulting in accurate reactive defenses. To better understand this phenomenon and reassess the reliability of the reactive defenses for medical AEs, we thoroughly investigate the characteristic of conventional medical AEs. Specifically, we first theoretically prove that conventional adversarial attacks change the outputs by continuously optimizing vulnerable features in a fixed direction, thereby leading to outlier representations in the feature space. Then, a stress test is conducted to reveal the vulnerability of medical images, by comparing with natural images. Interestingly, this vulnerability is a double-edged sword, which can be exploited to hide AEs. We then propose a simple-yet-effective hierarchical feature constraint (HFC), a novel add-on to conventional white-box attacks, which assists to hide the adversarial feature in the target feature distribution. The proposed method is evaluated on three medical datasets, both 2D and 3D, with different modalities. The experimental results demonstrate the superiority of HFC, \emph{i.e.,} it bypasses an array of state-of-the-art adversarial medical AE detectors more efficiently than competing adaptive attacks, which reveals the deficiencies of medical reactive defense and allows to develop more robust defenses in future.



## **5. The Queen's Guard: A Secure Enforcement of Fine-grained Access Control In Distributed Data Analytics Platforms**

cs.CR

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2106.13123v4) [paper-pdf](http://arxiv.org/pdf/2106.13123v4)

**Authors**: Fahad Shaon, Sazzadur Rahaman, Murat Kantarcioglu

**Abstract**: Distributed data analytics platforms (i.e., Apache Spark, Hadoop) provide high-level APIs to programmatically write analytics tasks that are run distributedly in multiple computing nodes. The design of these frameworks was primarily motivated by performance and usability. Thus, the security takes a back seat. Consequently, they do not inherently support fine-grained access control or offer any plugin mechanism to enable it, making them risky to be used in multi-tier organizational settings.   There have been attempts to build "add-on" solutions to enable fine-grained access control for distributed data analytics platforms. In this paper, first, we show that straightforward enforcement of ``add-on'' access control is insecure under adversarial code execution. Specifically, we show that an attacker can abuse platform-provided APIs to evade access controls without leaving any traces. Second, we designed a two-layered (i.e., proactive and reactive) defense system to protect against API abuses. On submission of a user code, our proactive security layer statically screens it to find potential attack signatures prior to its execution. The reactive security layer employs code instrumentation-based runtime checks and sandboxed execution to throttle any exploits at runtime. Next, we propose a new fine-grained access control framework with an enhanced policy language that supports map and filter primitives. Finally, we build a system named SecureDL with our new access control framework and defense system on top of Apache Spark, which ensures secure access control policy enforcement under adversaries capable of executing code.   To the best of our knowledge, this is the first fine-grained attribute-based access control framework for distributed data analytics platforms that is secure against platform API abuse attacks. Performance evaluation showed that the overhead due to added security is low.



## **6. Robust Evaluation of Diffusion-Based Adversarial Purification**

cs.CV

Accepted by ICCV 2023, oral presentation. Code is available at  https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2303.09051v3) [paper-pdf](http://arxiv.org/pdf/2303.09051v3)

**Authors**: Minjong Lee, Dongwoo Kim

**Abstract**: We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy improving robustness compared to the current diffusion-based purification methods.



## **7. Exploring Adversarial Robustness of LiDAR-Camera Fusion Model in Autonomous Driving**

cs.RO

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01468v1) [paper-pdf](http://arxiv.org/pdf/2312.01468v1)

**Authors**: Bo Yang, Xiaoyu Ji, Xiaoyu Ji, Xiaoyu Ji, Xiaoyu Ji

**Abstract**: Our study assesses the adversarial robustness of LiDAR-camera fusion models in 3D object detection. We introduce an attack technique that, by simply adding a limited number of physically constrained adversarial points above a car, can make the car undetectable by the fusion model. Experimental results reveal that even without changes to the image data channel, the fusion model can be deceived solely by manipulating the LiDAR data channel. This finding raises safety concerns in the field of autonomous driving. Further, we explore how the quantity of adversarial points, the distance between the front-near car and the LiDAR-equipped car, and various angular factors affect the attack success rate. We believe our research can contribute to the understanding of multi-sensor robustness, offering insights and guidance to enhance the safety of autonomous driving.



## **8. Evaluating the Security of Satellite Systems**

cs.CR

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01330v1) [paper-pdf](http://arxiv.org/pdf/2312.01330v1)

**Authors**: Roy Peled, Eran Aizikovich, Edan Habler, Yuval Elovici, Asaf Shabtai

**Abstract**: Satellite systems are facing an ever-increasing amount of cybersecurity threats as their role in communications, navigation, and other services expands. Recent papers have examined attacks targeting satellites and space systems; however, they did not comprehensively analyze the threats to satellites and systematically identify adversarial techniques across the attack lifecycle. This paper presents a comprehensive taxonomy of adversarial tactics, techniques, and procedures explicitly targeting LEO satellites. First, we analyze the space ecosystem including the ground, space, Communication, and user segments, highlighting their architectures, functions, and vulnerabilities. Then, we examine the threat landscape, including adversary types, and capabilities, and survey historical and recent attacks such as jamming, spoofing, and supply chain. Finally, we propose a novel extension of the MITRE ATT&CK framework to categorize satellite attack techniques across the adversary lifecycle from reconnaissance to impact. The taxonomy is demonstrated by modeling high-profile incidents, including the Viasat attack that disrupted Ukraine's communications. The taxonomy provides the foundation for the development of defenses against emerging cyber risks to space assets. The proposed threat model will advance research in the space domain and contribute to the security of the space domain against sophisticated attacks.



## **9. Rethinking PGD Attack: Is Sign Function Necessary?**

cs.LG

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01260v1) [paper-pdf](http://arxiv.org/pdf/2312.01260v1)

**Authors**: Junjie Yang, Tianlong Chen, Xuxi Chen, Zhangyang Wang, Yingbin Liang

**Abstract**: Neural networks have demonstrated success in various domains, yet their performance can be significantly degraded by even a small input perturbation. Consequently, the construction of such perturbations, known as adversarial attacks, has gained significant attention, many of which fall within "white-box" scenarios where we have full access to the neural network. Existing attack algorithms, such as the projected gradient descent (PGD), commonly take the sign function on the raw gradient before updating adversarial inputs, thereby neglecting gradient magnitude information. In this paper, we present a theoretical analysis of how such sign-based update algorithm influences step-wise attack performance, as well as its caveat. We also interpret why previous attempts of directly using raw gradients failed. Based on that, we further propose a new raw gradient descent (RGD) algorithm that eliminates the use of sign. Specifically, we convert the constrained optimization problem into an unconstrained one, by introducing a new hidden variable of non-clipped perturbation that can move beyond the constraint. The effectiveness of the proposed RGD algorithm has been demonstrated extensively in experiments, outperforming PGD and other competitors in various settings, without incurring any additional computational overhead. The codes is available in https://github.com/JunjieYang97/RGD.



## **10. Look Closer to Your Enemy: Learning to Attack via Teacher-Student Mimicking**

cs.CV

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2207.13381v4) [paper-pdf](http://arxiv.org/pdf/2207.13381v4)

**Authors**: Mingjie Wang, Jianxiong Guo, Sirui Li, Dingwen Xiao, Zhiqing Tang

**Abstract**: Deep neural networks have significantly advanced person re-identification (ReID) applications in the realm of the industrial internet, yet they remain vulnerable. Thus, it is crucial to study the robustness of ReID systems, as there are risks of adversaries using these vulnerabilities to compromise industrial surveillance systems. Current adversarial methods focus on generating attack samples using misclassification feedback from victim models (VMs), neglecting VM's cognitive processes. We seek to address this by producing authentic ReID attack instances through VM cognition decryption. This approach boasts advantages like better transferability to open-set ReID tests, easier VM misdirection, and enhanced creation of realistic and undetectable assault images. However, the task of deciphering the cognitive mechanism in VM is widely considered to be a formidable challenge. In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE (Look Closer to Your Enemy), to generate adversarial query images. Specifically, LCYE first distills VM's knowledge via teacher-student memory mimicking the proxy task. This knowledge prior serves as an unambiguous cryptographic token, encapsulating elements deemed indispensable and plausible by the VM, with the intent of facilitating precise adversarial misdirection. Further, benefiting from the multiple opposing task framework of LCYE, we investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. The source code can be found at https://github.com/MingjieWang0606/LCYE-attack_reid.



## **11. FRAUDability: Estimating Users' Susceptibility to Financial Fraud Using Adversarial Machine Learning**

cs.CR

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.01200v1) [paper-pdf](http://arxiv.org/pdf/2312.01200v1)

**Authors**: Chen Doytshman, Satoru Momiyama, Inderjeet Singh, Yuval Elovici, Asaf Shabtai

**Abstract**: In recent years, financial fraud detection systems have become very efficient at detecting fraud, which is a major threat faced by e-commerce platforms. Such systems often include machine learning-based algorithms aimed at detecting and reporting fraudulent activity. In this paper, we examine the application of adversarial learning based ranking techniques in the fraud detection domain and propose FRAUDability, a method for the estimation of a financial fraud detection system's performance for every user. We are motivated by the assumption that "not all users are created equal" -- while some users are well protected by fraud detection algorithms, others tend to pose a challenge to such systems. The proposed method produces scores, namely "fraudability scores," which are numerical estimations of a fraud detection system's ability to detect financial fraud for a specific user, given his/her unique activity in the financial system. Our fraudability scores enable those tasked with defending users in a financial platform to focus their attention and resources on users with high fraudability scores to better protect them. We validate our method using a real e-commerce platform's dataset and demonstrate the application of fraudability scores from the attacker's perspective, on the platform, and more specifically, on the fraud detection systems used by the e-commerce enterprise. We show that the scores can also help attackers increase their financial profit by 54%, by engaging solely with users with high fraudability scores, avoiding those users whose spending habits enable more accurate fraud detection.



## **12. Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions**

cs.LG

Published as a conference paper at NeurIPS 2023

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2306.15427v2) [paper-pdf](http://arxiv.org/pdf/2306.15427v2)

**Authors**: Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

**Abstract**: Despite its success in the image domain, adversarial training did not (yet) stand out as an effective defense for Graph Neural Networks (GNNs) against graph structure perturbations. In the pursuit of fixing adversarial training (1) we show and overcome fundamental theoretical as well as practical limitations of the adopted graph learning setting in prior work; (2) we reveal that more flexible GNNs based on learnable graph diffusion are able to adjust to adversarial perturbations, while the learned message passing scheme is naturally interpretable; (3) we introduce the first attack for structure perturbations that, while targeting multiple nodes at once, is capable of handling global (graph-level) as well as local (node-level) constraints. Including these contributions, we demonstrate that adversarial training is a state-of-the-art defense against adversarial structure perturbations.



## **13. Scrappy: SeCure Rate Assuring Protocol with PrivacY**

cs.CR

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.00989v1) [paper-pdf](http://arxiv.org/pdf/2312.00989v1)

**Authors**: Kosei Akama, Yoshimichi Nakatsuka, Masaaki Sato, Keisuke Uehara

**Abstract**: Preventing abusive activities caused by adversaries accessing online services at a rate exceeding that expected by websites has become an ever-increasing problem. CAPTCHAs and SMS authentication are widely used to provide a solution by implementing rate limiting, although they are becoming less effective, and some are considered privacy-invasive. In light of this, many studies have proposed better rate-limiting systems that protect the privacy of legitimate users while blocking malicious actors. However, they suffer from one or more shortcomings: (1) assume trust in the underlying hardware and (2) are vulnerable to side-channel attacks. Motivated by the aforementioned issues, this paper proposes Scrappy: SeCure Rate Assuring Protocol with PrivacY. Scrappy allows clients to generate unforgeable yet unlinkable rate-assuring proofs, which provides the server with cryptographic guarantees that the client is not misbehaving. We design Scrappy using a combination of DAA and hardware security devices. Scrappy is implemented over three types of devices, including one that can immediately be deployed in the real world. Our baseline evaluation shows that the end-to-end latency of Scrappy is minimal, taking only 0.32 seconds, and uses only 679 bytes of bandwidth when transferring necessary data. We also conduct an extensive security evaluation, showing that the rate-limiting capability of Scrappy is unaffected even if the hardware security device is compromised.



## **14. Deep Generative Attacks and Countermeasures for Data-Driven Offline Signature Verification**

cs.CV

10 pages, 7 figures, 1 table, Signature verification, Deep generative  models, attacks, generative attack explainability, data-driven verification  system

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.00987v1) [paper-pdf](http://arxiv.org/pdf/2312.00987v1)

**Authors**: An Ngo, MinhPhuong Cao, Rajesh Kumar

**Abstract**: While previous studies have explored attacks via random, simple, and skilled forgeries, generative attacks have received limited attention in the data-driven signature verification (DASV) process. Thus, this paper explores the impact of generative attacks on DASV and proposes practical and interpretable countermeasures. We investigate the power of two prominent Deep Generative Models (DGMs), Variational Auto-encoders (VAE) and Conditional Generative Adversarial Networks (CGAN), on their ability to generate signatures that would successfully deceive DASV. Additionally, we evaluate the quality of generated images using the Structural Similarity Index measure (SSIM) and use the same to explain the attack's success. Finally, we propose countermeasures that effectively reduce the impact of deep generative attacks on DASV.   We first generated six synthetic datasets from three benchmark offline-signature datasets viz. CEDAR, BHSig260- Bengali, and BHSig260-Hindi using VAE and CGAN. Then, we built baseline DASVs using Xception, ResNet152V2, and DenseNet201. These DASVs achieved average (over the three datasets) False Accept Rates (FARs) of 2.55%, 3.17%, and 1.06%, respectively. Then, we attacked these baselines using the synthetic datasets. The VAE-generated signatures increased average FARs to 10.4%, 10.1%, and 7.5%, while CGAN-generated signatures to 32.5%, 30%, and 26.1%. The variation in the effectiveness of attack for VAE and CGAN was investigated further and explained by a strong (rho = -0.86) negative correlation between FARs and SSIMs. We created another set of synthetic datasets and used the same to retrain the DASVs. The retained baseline showed significant robustness to random, skilled, and generative attacks as the FARs shrank to less than 1% on average. The findings underscore the importance of studying generative attacks and potential countermeasures for DASV.



## **15. Stealing the Decoding Algorithms of Language Models**

cs.LG

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2303.04729v4) [paper-pdf](http://arxiv.org/pdf/2303.04729v4)

**Authors**: Ali Naseh, Kalpesh Krishna, Mohit Iyyer, Amir Houmansadr

**Abstract**: A key component of generating text from modern language models (LM) is the selection and tuning of decoding algorithms. These algorithms determine how to generate text from the internal probability distribution generated by the LM. The process of choosing a decoding algorithm and tuning its hyperparameters takes significant time, manual effort, and computation, and it also requires extensive human evaluation. Therefore, the identity and hyperparameters of such decoding algorithms are considered to be extremely valuable to their owners. In this work, we show, for the first time, that an adversary with typical API access to an LM can steal the type and hyperparameters of its decoding algorithms at very low monetary costs. Our attack is effective against popular LMs used in text generation APIs, including GPT-2, GPT-3 and GPT-Neo. We demonstrate the feasibility of stealing such information with only a few dollars, e.g., $\$0.8$, $\$1$, $\$4$, and $\$40$ for the four versions of GPT-3.



## **16. Adversarial Attacks and Defenses on 3D Point Cloud Classification: A Survey**

cs.CV

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2307.00309v2) [paper-pdf](http://arxiv.org/pdf/2307.00309v2)

**Authors**: Hanieh Naderi, Ivan V. Bajić

**Abstract**: Deep learning has successfully solved a wide range of tasks in 2D vision as a dominant AI technique. Recently, deep learning on 3D point clouds is becoming increasingly popular for addressing various tasks in this field. Despite remarkable achievements, deep learning algorithms are vulnerable to adversarial attacks. These attacks are imperceptible to the human eye but can easily fool deep neural networks in the testing and deployment stage. To encourage future research, this survey summarizes the current progress on adversarial attack and defense techniques on point cloud classification.This paper first introduces the principles and characteristics of adversarial attacks and summarizes and analyzes adversarial example generation methods in recent years. Additionally, it provides an overview of defense strategies, organized into data-focused and model-focused methods. Finally, it presents several current challenges and potential future research directions in this domain.



## **17. A Unified Approach to Interpreting and Boosting Adversarial Transferability**

cs.LG

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2010.04055v2) [paper-pdf](http://arxiv.org/pdf/2010.04055v2)

**Authors**: Xin Wang, Jie Ren, Shuyun Lin, Xiangming Zhu, Yisen Wang, Quanshi Zhang

**Abstract**: In this paper, we use the interaction inside adversarial perturbations to explain and boost the adversarial transferability. We discover and prove the negative correlation between the adversarial transferability and the interaction inside adversarial perturbations. The negative correlation is further verified through different DNNs with various inputs. Moreover, this negative correlation can be regarded as a unified perspective to understand current transferability-boosting methods. To this end, we prove that some classic methods of enhancing the transferability essentially decease interactions inside adversarial perturbations. Based on this, we propose to directly penalize interactions during the attacking process, which significantly improves the adversarial transferability.



## **18. PyraTrans: Learning Attention-Enriched Multi-Scale Pyramid Network from Pre-Trained Transformers for Effective Malicious URL Detection**

cs.CR

12 pages, 7 figures

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2312.00508v1) [paper-pdf](http://arxiv.org/pdf/2312.00508v1)

**Authors**: Ruitong Liu, Yanbin Wang, Zhenhao Guo, Haitao Xu, Zhan Qin, Wenrui Ma, Fan Zhang

**Abstract**: Detecting malicious URLs is a crucial aspect of web search and mining, significantly impacting internet security. Though advancements in machine learning have improved the effectiveness of detection methods, these methods still face significant challenges in their capacity to generalize and their resilience against evolving threats. In this paper, we propose PyraTrans, an approach that combines the strengths of pretrained Transformers and pyramid feature learning for improving malicious URL detection. We implement PyraTrans by leveraging a pretrained CharBERT as the base and augmenting it with 3 connected feature modules: 1) The Encoder Feature Extraction module, which extracts representations from each encoder layer of CharBERT to obtain multi-order features; 2) The Multi-Scale Feature Learning Module, which captures multi-scale local contextual insights and aggregate information across different layer-levels; and 3) The Pyramid Spatial Attention Module, which learns hierarchical and spatial feature attentions, highlighting critical classification signals while reducing noise. The proposed approach addresses the limitations of the Transformer in local feature learning and spatial awareness, and enabling us to extract multi-order, multi-scale URL feature representations with enhanced attentional focus. PyraTrans is evaluated using 4 benchmark datasets, where it demonstrated significant advancements over prior baseline methods. Particularly, on the imbalanced dataset, our method, with just 10% of the data for training, the TPR is 3.3-6.5 times and the F1-score is 2.9-4.5 times that of the baseline. Our approach also demonstrates robustness against adversarial attacks. Codes and data are available at https://github.com/Alixyvtte/PyraTrans.



## **19. Unleashing Cheapfakes through Trojan Plugins of Large Language Models**

cs.CR

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2312.00374v1) [paper-pdf](http://arxiv.org/pdf/2312.00374v1)

**Authors**: Tian Dong, Guoxing Chen, Shaofeng Li, Minhui Xue, Rayne Holland, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. Our experiments validate that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserves or improves the adapter's utility. Finally, we provide two case studies to demonstrate that the Trojan adapter can lead a LLM-powered autonomous agent to execute unintended scripts or send phishing emails. Our novel attacks represent the first study of supply chain threats for LLMs through the lens of Trojan plugins.



## **20. Bayesian Learning with Information Gain Provably Bounds Risk for a Robust Adversarial Defense**

cs.LG

Published at ICML 2022. Code is available at  https://github.com/baogiadoan/IG-BNN

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2212.02003v2) [paper-pdf](http://arxiv.org/pdf/2212.02003v2)

**Authors**: Bao Gia Doan, Ehsan Abbasnejad, Javen Qinfeng Shi, Damith C. Ranasinghe

**Abstract**: We present a new algorithm to learn a deep neural network model robust against adversarial attacks. Previous algorithms demonstrate an adversarially trained Bayesian Neural Network (BNN) provides improved robustness. We recognize the adversarial learning approach for approximating the multi-modal posterior distribution of a Bayesian model can lead to mode collapse; consequently, the model's achievements in robustness and performance are sub-optimal. Instead, we first propose preventing mode collapse to better approximate the multi-modal posterior distribution. Second, based on the intuition that a robust model should ignore perturbations and only consider the informative content of the input, we conceptualize and formulate an information gain objective to measure and force the information learned from both benign and adversarial training instances to be similar. Importantly. we prove and demonstrate that minimizing the information gain objective allows the adversarial risk to approach the conventional empirical risk. We believe our efforts provide a step toward a basis for a principled method of adversarially training BNNs. Our model demonstrate significantly improved robustness--up to 20%--compared with adversarial training and Adv-BNN under PGD attacks with 0.035 distortion on both CIFAR-10 and STL-10 datasets.



## **21. Security Defense of Large Scale Networks Under False Data Injection Attacks: An Attack Detection Scheduling Approach**

eess.SY

14 pages, 11 figures

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2212.05500v3) [paper-pdf](http://arxiv.org/pdf/2212.05500v3)

**Authors**: Yuhan Suo, Senchun Chai, Runqi Chai, Zhong-Hua Pang, Yuanqing Xia, Guo-Ping Liu

**Abstract**: In large-scale networks, communication links between nodes are easily injected with false data by adversaries. This paper proposes a novel security defense strategy from the perspective of attack detection scheduling to ensure the security of the network. Based on the proposed strategy, each sensor can directly exclude suspicious sensors from its neighboring set. First, the problem of selecting suspicious sensors is formulated as a combinatorial optimization problem, which is non-deterministic polynomial-time hard (NP-hard). To solve this problem, the original function is transformed into a submodular function. Then, we propose a distributed attack detection scheduling algorithm based on the sequential submodular optimization theory, which incorporates \emph{expert problem} to better utilize historical information to guide the sensor selection task at the current moment. For different attack strategies, theoretical results show that the average optimization rate of the proposed algorithm has a lower bound, and the error expectation for any subset is bounded. In addition, under two kinds of insecurity conditions, the proposed algorithm can guarantee the security of the entire network from the perspective of the augmented estimation error. Finally, the effectiveness of the proposed method is verified by the numerical simulation and practical experiment.



## **22. SPAM: Secure & Private Aircraft Management**

cs.CR

6 pages

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00245v1) [paper-pdf](http://arxiv.org/pdf/2312.00245v1)

**Authors**: Yaman Jandali, Nojan Sheybani, Farinaz Koushanfar

**Abstract**: With the rising use of aircrafts for operations ranging from disaster-relief to warfare, there is a growing risk of adversarial attacks. Malicious entities often only require the location of the aircraft for these attacks. Current satellite-aircraft communication and tracking protocols put aircrafts at risk if the satellite is compromised, due to computation being done in plaintext. In this work, we present \texttt{SPAM}, a private, secure, and accurate system that allows satellites to efficiently manage and maintain tracking angles for aircraft fleets without learning aircrafts' locations. \texttt{SPAM} is built upon multi-party computation and zero-knowledge proofs to guarantee privacy and high efficiency. While catered towards aircrafts, \texttt{SPAM}'s zero-knowledge fleet management can be easily extended to the IoT, with very little overhead.



## **23. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16119v2) [paper-pdf](http://arxiv.org/pdf/2311.16119v2)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.



## **24. Optimal Attack and Defense for Reinforcement Learning**

cs.LG

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00198v1) [paper-pdf](http://arxiv.org/pdf/2312.00198v1)

**Authors**: Jeremy McMahan, Young Wu, Xiaojin Zhu, Qiaomin Xie

**Abstract**: To ensure the usefulness of Reinforcement Learning (RL) in real systems, it is crucial to ensure they are robust to noise and adversarial attacks. In adversarial RL, an external attacker has the power to manipulate the victim agent's interaction with the environment. We study the full class of online manipulation attacks, which include (i) state attacks, (ii) observation attacks (which are a generalization of perceived-state attacks), (iii) action attacks, and (iv) reward attacks. We show the attacker's problem of designing a stealthy attack that maximizes its own expected reward, which often corresponds to minimizing the victim's value, is captured by a Markov Decision Process (MDP) that we call a meta-MDP since it is not the true environment but a higher level environment induced by the attacked interaction. We show that the attacker can derive optimal attacks by planning in polynomial time or learning with polynomial sample complexity using standard RL techniques. We argue that the optimal defense policy for the victim can be computed as the solution to a stochastic Stackelberg game, which can be further simplified into a partially-observable turn-based stochastic game (POTBSG). Neither the attacker nor the victim would benefit from deviating from their respective optimal policies, thus such solutions are truly robust. Although the defense problem is NP-hard, we show that optimal Markovian defenses can be computed (learned) in polynomial time (sample complexity) in many scenarios.



## **25. Fool the Hydra: Adversarial Attacks against Multi-view Object Detection Systems**

cs.CV

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00173v1) [paper-pdf](http://arxiv.org/pdf/2312.00173v1)

**Authors**: Bilel Tarchoun, Quazi Mishkatul Alam, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstract**: Adversarial patches exemplify the tangible manifestation of the threat posed by adversarial attacks on Machine Learning (ML) models in real-world scenarios. Robustness against these attacks is of the utmost importance when designing computer vision applications, especially for safety-critical domains such as CCTV systems. In most practical situations, monitoring open spaces requires multi-view systems to overcome acquisition challenges such as occlusion handling. Multiview object systems are able to combine data from multiple views, and reach reliable detection results even in difficult environments. Despite its importance in real-world vision applications, the vulnerability of multiview systems to adversarial patches is not sufficiently investigated. In this paper, we raise the following question: Does the increased performance and information sharing across views offer as a by-product robustness to adversarial patches? We first conduct a preliminary analysis showing promising robustness against off-the-shelf adversarial patches, even in an extreme setting where we consider patches applied to all views by all persons in Wildtrack benchmark. However, we challenged this observation by proposing two new attacks: (i) In the first attack, targeting a multiview CNN, we maximize the global loss by proposing gradient projection to the different views and aggregating the obtained local gradients. (ii) In the second attack, we focus on a Transformer-based multiview framework. In addition to the focal loss, we also maximize the transformer-specific loss by dissipating its attention blocks. Our results show a large degradation in the detection performance of victim multiview systems with our first patch attack reaching an attack success rate of 73% , while our second proposed attack reduced the performance of its target detector by 62%



## **26. Universal Backdoor Attacks**

cs.LG

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00157v1) [paper-pdf](http://arxiv.org/pdf/2312.00157v1)

**Authors**: Benjamin Schneider, Nils Lukas, Florian Kerschbaum

**Abstract**: Web-scraped datasets are vulnerable to data poisoning, which can be used for backdooring deep image classifiers during training. Since training on large datasets is expensive, a model is trained once and re-used many times. Unlike adversarial examples, backdoor attacks often target specific classes rather than any class learned by the model. One might expect that targeting many classes through a naive composition of attacks vastly increases the number of poison samples. We show this is not necessarily true and more efficient, universal data poisoning attacks exist that allow controlling misclassifications from any source class into any target class with a small increase in poison samples. Our idea is to generate triggers with salient characteristics that the model can learn. The triggers we craft exploit a phenomenon we call inter-class poison transferability, where learning a trigger from one class makes the model more vulnerable to learning triggers for other classes. We demonstrate the effectiveness and robustness of our universal backdoor attacks by controlling models with up to 6,000 classes while poisoning only 0.15% of the training dataset.



## **27. On the Adversarial Robustness of Graph Contrastive Learning Methods**

cs.LG

Accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop  (NeurIPS GLFrontiers 2023)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.17853v2) [paper-pdf](http://arxiv.org/pdf/2311.17853v2)

**Authors**: Filippo Guerranti, Zinuo Yi, Anna Starovoit, Rafiq Kamel, Simon Geisler, Stephan Günnemann

**Abstract**: Contrastive learning (CL) has emerged as a powerful framework for learning representations of images and text in a self-supervised manner while enhancing model robustness against adversarial attacks. More recently, researchers have extended the principles of contrastive learning to graph-structured data, giving birth to the field of graph contrastive learning (GCL). However, whether GCL methods can deliver the same advantages in adversarial robustness as their counterparts in the image and text domains remains an open question. In this paper, we introduce a comprehensive robustness evaluation protocol tailored to assess the robustness of GCL models. We subject these models to adaptive adversarial attacks targeting the graph structure, specifically in the evasion scenario. We evaluate node and graph classification tasks using diverse real-world datasets and attack strategies. With our work, we aim to offer insights into the robustness of GCL methods and hope to open avenues for potential future research directions.



## **28. Adversarial Attacks and Defenses for Wireless Signal Classifiers using CDI-aware GANs**

cs.IT

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.18820v1) [paper-pdf](http://arxiv.org/pdf/2311.18820v1)

**Authors**: Sujata Sinha, Alkan Soysal

**Abstract**: We introduce a Channel Distribution Information (CDI)-aware Generative Adversarial Network (GAN), designed to address the unique challenges of adversarial attacks in wireless communication systems. The generator in this CDI-aware GAN maps random input noise to the feature space, generating perturbations intended to deceive a target modulation classifier. Its discriminators play a dual role: one enforces that the perturbations follow a Gaussian distribution, making them indistinguishable from Gaussian noise, while the other ensures these perturbations account for realistic channel effects and resemble no-channel perturbations.   Our proposed CDI-aware GAN can be used as an attacker and a defender. In attack scenarios, the CDI-aware GAN demonstrates its prowess by generating robust adversarial perturbations that effectively deceive the target classifier, outperforming known methods. Furthermore, CDI-aware GAN as a defender significantly improves the target classifier's resilience against adversarial attacks.



## **29. Improving the Robustness of Quantized Deep Neural Networks to White-Box Attacks using Stochastic Quantization and Information-Theoretic Ensemble Training**

cs.CV

9 pages, 9 figures, 4 tables

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00105v1) [paper-pdf](http://arxiv.org/pdf/2312.00105v1)

**Authors**: Saurabh Farkya, Aswin Raghavan, Avi Ziskind

**Abstract**: Most real-world applications that employ deep neural networks (DNNs) quantize them to low precision to reduce the compute needs. We present a method to improve the robustness of quantized DNNs to white-box adversarial attacks. We first tackle the limitation of deterministic quantization to fixed ``bins'' by introducing a differentiable Stochastic Quantizer (SQ). We explore the hypothesis that different quantizations may collectively be more robust than each quantized DNN. We formulate a training objective to encourage different quantized DNNs to learn different representations of the input image. The training objective captures diversity and accuracy via mutual information between ensemble members. Through experimentation, we demonstrate substantial improvement in robustness against $L_\infty$ attacks even if the attacker is allowed to backpropagate through SQ (e.g., > 50\% accuracy to PGD(5/255) on CIFAR10 without adversarial training), compared to vanilla DNNs as well as existing ensembles of quantized DNNs. We extend the method to detect attacks and generate robustness profiles in the adversarial information plane (AIP), towards a unified analysis of different threat models by correlating the MI and accuracy.



## **30. Differentiable JPEG: The Devil is in the Details**

cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2309.06978v3) [paper-pdf](http://arxiv.org/pdf/2309.06978v3)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.



## **31. Diffusion Models for Imperceptible and Transferable Adversarial Attack**

cs.CV

Code Page: https://github.com/WindVChen/DiffAttack. In Paper Version  v2, we incorporate more discussions and experiments

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2305.08192v2) [paper-pdf](http://arxiv.org/pdf/2305.08192v2)

**Authors**: Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi

**Abstract**: Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models. Specifically, instead of direct manipulation in pixel space, we craft perturbations in the latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an implicit recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, DiffAttack, is the first that introduces diffusion models into the adversarial attack field. Extensive experiments on various model structures, datasets, and defense methods have demonstrated the superiority of our attack over the existing attack methods.



## **32. Data-Agnostic Model Poisoning against Federated Learning: A Graph Autoencoder Approach**

cs.LG

15 pages, 10 figures, submitted to IEEE Transactions on Information  Forensics and Security (TIFS)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.18498v1) [paper-pdf](http://arxiv.org/pdf/2311.18498v1)

**Authors**: Kai Li, Jingjing Zheng, Xin Yuan, Wei Ni, Ozgur B. Akan, H. Vincent Poor

**Abstract**: This paper proposes a novel, data-agnostic, model poisoning attack on Federated Learning (FL), by designing a new adversarial graph autoencoder (GAE)-based framework. The attack requires no knowledge of FL training data and achieves both effectiveness and undetectability. By listening to the benign local models and the global model, the attacker extracts the graph structural correlations among the benign local models and the training data features substantiating the models. The attacker then adversarially regenerates the graph structural correlations while maximizing the FL training loss, and subsequently generates malicious local models using the adversarial graph structure and the training data features of the benign ones. A new algorithm is designed to iteratively train the malicious local models using GAE and sub-gradient descent. The convergence of FL under attack is rigorously proved, with a considerably large optimality gap. Experiments show that the FL accuracy drops gradually under the proposed attack and existing defense mechanisms fail to detect it. The attack can give rise to an infection across all benign devices, making it a serious threat to FL.



## **33. Towards Safer Generative Language Models: A Survey on Safety Risks, Evaluations, and Improvements**

cs.AI

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2302.09270v3) [paper-pdf](http://arxiv.org/pdf/2302.09270v3)

**Authors**: Jiawen Deng, Jiale Cheng, Hao Sun, Zhexin Zhang, Minlie Huang

**Abstract**: As generative large model capabilities advance, safety concerns become more pronounced in their outputs. To ensure the sustainable growth of the AI ecosystem, it's imperative to undertake a holistic evaluation and refinement of associated safety risks. This survey presents a framework for safety research pertaining to large models, delineating the landscape of safety risks as well as safety evaluation and improvement methods. We begin by introducing safety issues of wide concern, then delve into safety evaluation methods for large models, encompassing preference-based testing, adversarial attack approaches, issues detection, and other advanced evaluation methods. Additionally, we explore the strategies for enhancing large model safety from training to deployment, highlighting cutting-edge safety approaches for each stage in building large models. Finally, we discuss the core challenges in advancing towards more responsible AI, including the interpretability of safety mechanisms, ongoing safety issues, and robustness against malicious attacks. Through this survey, we aim to provide clear technical guidance for safety researchers and encourage further study on the safety of large models.



## **34. On the Robustness of Decision-Focused Learning**

cs.LG

17 pages, 45 figures, submitted to AAAI artificial intelligence for  operations research workshop

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16487v2) [paper-pdf](http://arxiv.org/pdf/2311.16487v2)

**Authors**: Yehya Farhat

**Abstract**: Decision-Focused Learning (DFL) is an emerging learning paradigm that tackles the task of training a machine learning (ML) model to predict missing parameters of an incomplete optimization problem, where the missing parameters are predicted. DFL trains an ML model in an end-to-end system, by integrating the prediction and optimization tasks, providing better alignment of the training and testing objectives. DFL has shown a lot of promise and holds the capacity to revolutionize decision-making in many real-world applications. However, very little is known about the performance of these models under adversarial attacks. We adopt ten unique DFL methods and benchmark their performance under two distinctly focused attacks adapted towards the Predict-then-Optimize problem setting. Our study proposes the hypothesis that the robustness of a model is highly correlated with its ability to find predictions that lead to optimal decisions without deviating from the ground-truth label. Furthermore, we provide insight into how to target the models that violate this condition and show how these models respond differently depending on the achieved optimality at the end of their training cycles.



## **35. Improving the Robustness of Transformer-based Large Language Models with Dynamic Attention**

cs.CL

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.17400v2) [paper-pdf](http://arxiv.org/pdf/2311.17400v2)

**Authors**: Lujia Shen, Yuwen Pu, Shouling Ji, Changjiang Li, Xuhong Zhang, Chunpeng Ge, Ting Wang

**Abstract**: Transformer-based models, such as BERT and GPT, have been widely adopted in natural language processing (NLP) due to their exceptional performance. However, recent studies show their vulnerability to textual adversarial attacks where the model's output can be misled by intentionally manipulating the text inputs. Despite various methods that have been proposed to enhance the model's robustness and mitigate this vulnerability, many require heavy consumption resources (e.g., adversarial training) or only provide limited protection (e.g., defensive dropout). In this paper, we propose a novel method called dynamic attention, tailored for the transformer architecture, to enhance the inherent robustness of the model itself against various adversarial attacks. Our method requires no downstream task knowledge and does not incur additional costs. The proposed dynamic attention consists of two modules: (I) attention rectification, which masks or weakens the attention value of the chosen tokens, and (ii) dynamic modeling, which dynamically builds the set of candidate tokens. Extensive experiments demonstrate that dynamic attention significantly mitigates the impact of adversarial attacks, improving up to 33\% better performance than previous methods against widely-used adversarial attacks. The model-level design of dynamic attention enables it to be easily combined with other defense methods (e.g., adversarial training) to further enhance the model's robustness. Furthermore, we demonstrate that dynamic attention preserves the state-of-the-art robustness space of the original model compared to other dynamic modeling methods.



## **36. Effective Backdoor Mitigation Depends on the Pre-training Objective**

cs.LG

Accepted for oral presentation at BUGS workshop @ NeurIPS 2023  (https://neurips2023-bugs.github.io/)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.14948v2) [paper-pdf](http://arxiv.org/pdf/2311.14948v2)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for pre-training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in these models such as CleanCLIP which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP is ineffective when stronger pre-training objectives are used, even with extensive hyperparameter tuning. Our findings underscore critical considerations for ML practitioners who pre-train models using large-scale web-curated data and are concerned about potential backdoor threats. Notably, our results suggest that simpler pre-training objectives are more amenable to effective backdoor removal. This insight is pivotal for practitioners seeking to balance the trade-offs between using stronger pre-training objectives and security against backdoor attacks.



## **37. AnonPSI: An Anonymity Assessment Framework for PSI**

cs.CR

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.18118v1) [paper-pdf](http://arxiv.org/pdf/2311.18118v1)

**Authors**: Bo Jiang, Jian Du, Qiang Yan

**Abstract**: Private Set Intersection (PSI) is a widely used protocol that enables two parties to securely compute a function over the intersected part of their shared datasets and has been a significant research focus over the years. However, recent studies have highlighted its vulnerability to Set Membership Inference Attacks (SMIA), where an adversary might deduce an individual's membership by invoking multiple PSI protocols. This presents a considerable risk, even in the most stringent versions of PSI, which only return the cardinality of the intersection. This paper explores the evaluation of anonymity within the PSI context. Initially, we highlight the reasons why existing works fall short in measuring privacy leakage, and subsequently propose two attack strategies that address these deficiencies. Furthermore, we provide theoretical guarantees on the performance of our proposed methods. In addition to these, we illustrate how the integration of auxiliary information, such as the sum of payloads associated with members of the intersection (PSI-SUM), can enhance attack efficiency. We conducted a comprehensive performance evaluation of various attack strategies proposed utilizing two real datasets. Our findings indicate that the methods we propose markedly enhance attack efficiency when contrasted with previous research endeavors. {The effective attacking implies that depending solely on existing PSI protocols may not provide an adequate level of privacy assurance. It is recommended to combine privacy-enhancing technologies synergistically to enhance privacy protection even further.



## **38. Improving Faithfulness for Vision Transformers**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17983v1) [paper-pdf](http://arxiv.org/pdf/2311.17983v1)

**Authors**: Lijie Hu, Yixin Liu, Ninghao Liu, Mengdi Huai, Lichao Sun, Di Wang

**Abstract**: Vision Transformers (ViTs) have achieved state-of-the-art performance for various vision tasks. One reason behind the success lies in their ability to provide plausible innate explanations for the behavior of neural architectures. However, ViTs suffer from issues with explanation faithfulness, as their focal points are fragile to adversarial attacks and can be easily changed with even slight perturbations on the input image. In this paper, we propose a rigorous approach to mitigate these issues by introducing Faithful ViTs (FViTs). Briefly speaking, an FViT should have the following two properties: (1) The top-$k$ indices of its self-attention vector should remain mostly unchanged under input perturbation, indicating stable explanations; (2) The prediction distribution should be robust to perturbations. To achieve this, we propose a new method called Denoised Diffusion Smoothing (DDS), which adopts randomized smoothing and diffusion-based denoising. We theoretically prove that processing ViTs directly with DDS can turn them into FViTs. We also show that Gaussian noise is nearly optimal for both $\ell_2$ and $\ell_\infty$-norm cases. Finally, we demonstrate the effectiveness of our approach through comprehensive experiments and evaluations. Specifically, we compare our FViTs with other baselines through visual interpretation and robustness accuracy under adversarial attacks. Results show that FViTs are more robust against adversarial attacks while maintaining the explainability of attention, indicating higher faithfulness.



## **39. SenTest: Evaluating Robustness of Sentence Encoders**

cs.CL

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17722v1) [paper-pdf](http://arxiv.org/pdf/2311.17722v1)

**Authors**: Tanmay Chavan, Shantanu Patankar, Aditya Kane, Omkar Gokhale, Geetanjali Kale, Raviraj Joshi

**Abstract**: Contrastive learning has proven to be an effective method for pre-training models using weakly labeled data in the vision domain. Sentence transformers are the NLP counterparts to this architecture, and have been growing in popularity due to their rich and effective sentence representations. Having effective sentence representations is paramount in multiple tasks, such as information retrieval, retrieval augmented generation (RAG), and sentence comparison. Keeping in mind the deployability factor of transformers, evaluating the robustness of sentence transformers is of utmost importance. This work focuses on evaluating the robustness of the sentence encoders. We employ several adversarial attacks to evaluate its robustness. This system uses character-level attacks in the form of random character substitution, word-level attacks in the form of synonym replacement, and sentence-level attacks in the form of intra-sentence word order shuffling. The results of the experiments strongly undermine the robustness of sentence encoders. The models produce significantly different predictions as well as embeddings on perturbed datasets. The accuracy of the models can fall up to 15 percent on perturbed datasets as compared to unperturbed datasets. Furthermore, the experiments demonstrate that these embeddings does capture the semantic and syntactic structure (sentence order) of sentences. However, existing supervised classification strategies fail to leverage this information, and merely function as n-gram detectors.



## **40. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

cs.LG

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2310.03684v3) [paper-pdf](http://arxiv.org/pdf/2310.03684v3)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM. Our code is publicly available at the following link: https://github.com/arobey1/smooth-llm.



## **41. Natural & Adversarial Bokeh Rendering via Circle-of-Confusion Predictive Network**

cs.CV

11 pages, accepted by TMM

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2111.12971v3) [paper-pdf](http://arxiv.org/pdf/2111.12971v3)

**Authors**: Yihao Huang, Felix Juefei-Xu, Qing Guo, Geguang Pu, Yang Liu

**Abstract**: Bokeh effect is a natural shallow depth-of-field phenomenon that blurs the out-of-focus part in photography. In recent years, a series of works have proposed automatic and realistic bokeh rendering methods for artistic and aesthetic purposes. They usually employ cutting-edge data-driven deep generative networks with complex training strategies and network architectures. However, these works neglect that the bokeh effect, as a real phenomenon, can inevitably affect the subsequent visual intelligent tasks like recognition, and their data-driven nature prevents them from studying the influence of bokeh-related physical parameters (i.e., depth-of-the-field) on the intelligent tasks. To fill this gap, we study a totally new problem, i.e., natural & adversarial bokeh rendering, which consists of two objectives: rendering realistic and natural bokeh and fooling the visual perception models (i.e., bokeh-based adversarial attack). To this end, beyond the pure data-driven solution, we propose a hybrid alternative by taking the respective advantages of data-driven and physical-aware methods. Specifically, we propose the circle-of-confusion predictive network (CoCNet) by taking the all-in-focus image and depth image as inputs to estimate circle-of-confusion parameters for each pixel, which are employed to render the final image through a well-known physical model of bokeh. With the hybrid solution, our method could achieve more realistic rendering results with the naive training strategy and a much lighter network.



## **42. Query-Relevant Images Jailbreak Large Multi-Modal Models**

cs.CV

Technique report

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17600v1) [paper-pdf](http://arxiv.org/pdf/2311.17600v1)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Warning: This paper contains examples of harmful language and images, and reader discretion is recommended. The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Large Multi-Modal Models (LMMs) remains understudied. In our study, we present a novel visual prompt attack that exploits query-relevant images to jailbreak the open-source LMMs. Our method creates a composite image from one image generated by diffusion models and another that displays the text as typography, based on keywords extracted from a malicious query. We show LLMs can be easily attacked by our approach, even if the employed Large Language Models are safely aligned. To evaluate the extent of this vulnerability in open-source LMMs, we have compiled a substantial dataset encompassing 13 scenarios with a total of 5,040 text-image pairs, using our presented attack technique. Our evaluation of 12 cutting-edge LMMs using this dataset shows the vulnerability of existing multi-modal models on adversarial attacks. This finding underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source LMMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **43. Quantum Neural Networks under Depolarization Noise: Exploring White-Box Attacks and Defenses**

quant-ph

Poster at Quantum Techniques in Machine Learning (QTML) 2023

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17458v1) [paper-pdf](http://arxiv.org/pdf/2311.17458v1)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: Leveraging the unique properties of quantum mechanics, Quantum Machine Learning (QML) promises computational breakthroughs and enriched perspectives where traditional systems reach their boundaries. However, similarly to classical machine learning, QML is not immune to adversarial attacks. Quantum adversarial machine learning has become instrumental in highlighting the weak points of QML models when faced with adversarial crafted feature vectors. Diving deep into this domain, our exploration shines light on the interplay between depolarization noise and adversarial robustness. While previous results enhanced robustness from adversarial threats through depolarization noise, our findings paint a different picture. Interestingly, adding depolarization noise discontinued the effect of providing further robustness for a multi-class classification scenario. Consolidating our findings, we conducted experiments with a multi-class classifier adversarially trained on gate-based quantum simulators, further elucidating this unexpected behavior.



## **44. Group-wise Sparse and Explainable Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17434v1) [paper-pdf](http://arxiv.org/pdf/2311.17434v1)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, typically regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs than previously anticipated. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. In this paper, we tackle this challenge by presenting an algorithm that simultaneously generates group-wise sparse attacks within semantically meaningful areas of an image. In each iteration, the core operation of our algorithm involves the optimization of a quasinorm adversarial loss. This optimization is achieved by employing the $1/2$-quasinorm proximal operator for some iterations, a method tailored for nonconvex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2$-norm regularization applied to perturbation magnitudes. We rigorously evaluate the efficacy of our novel attack in both targeted and non-targeted attack scenarios, on CIFAR-10 and ImageNet datasets. When compared to state-of-the-art methods, our attack consistently results in a remarkable increase in group-wise sparsity, e.g., an increase of $48.12\%$ on CIFAR-10 and $40.78\%$ on ImageNet (average case, targeted attack), all while maintaining lower perturbation magnitudes. Notably, this performance is complemented by a significantly faster computation time and a $100\%$ attack success rate.



## **45. Enhancing Adversarial Attacks: The Similar Target Method**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2308.10743v3) [paper-pdf](http://arxiv.org/pdf/2308.10743v3)

**Authors**: Shuo Zhang, Ziruo Wang, Zikai Zhou, Huanran Chen

**Abstract**: Deep neural networks are vulnerable to adversarial examples, posing a threat to the models' applications and raising security concerns. An intriguing property of adversarial examples is their strong transferability. Several methods have been proposed to enhance transferability, including ensemble attacks which have demonstrated their efficacy. However, prior approaches simply average logits, probabilities, or losses for model ensembling, lacking a comprehensive analysis of how and why model ensembling significantly improves transferability. In this paper, we propose a similar targeted attack method named Similar Target~(ST). By promoting cosine similarity between the gradients of each model, our method regularizes the optimization direction to simultaneously attack all surrogate models. This strategy has been proven to enhance generalization ability. Experimental results on ImageNet validate the effectiveness of our approach in improving adversarial transferability. Our method outperforms state-of-the-art attackers on 18 discriminative classifiers and adversarially trained models.



## **46. RADAP: A Robust and Adaptive Defense Against Diverse Adversarial Patches on Face Recognition**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17339v1) [paper-pdf](http://arxiv.org/pdf/2311.17339v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao, Changhai Nie

**Abstract**: Face recognition (FR) systems powered by deep learning have become widely used in various applications. However, they are vulnerable to adversarial attacks, especially those based on local adversarial patches that can be physically applied to real-world objects. In this paper, we propose RADAP, a robust and adaptive defense mechanism against diverse adversarial patches in both closed-set and open-set FR systems. RADAP employs innovative techniques, such as FCutout and F-patch, which use Fourier space sampling masks to improve the occlusion robustness of the FR model and the performance of the patch segmenter. Moreover, we introduce an edge-aware binary cross-entropy (EBCE) loss function to enhance the accuracy of patch detection. We also present the split and fill (SAF) strategy, which is designed to counter the vulnerability of the patch segmenter to complete white-box adaptive attacks. We conduct comprehensive experiments to validate the effectiveness of RADAP, which shows significant improvements in defense performance against various adversarial patches, while maintaining clean accuracy higher than that of the undefended Vanilla model.



## **47. NeRFTAP: Enhancing Transferability of Adversarial Patches on Face Recognition using Neural Radiance Fields**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17332v1) [paper-pdf](http://arxiv.org/pdf/2311.17332v1)

**Authors**: Xiaoliang Liu, Furao Shen, Feng Han, Jian Zhao, Changhai Nie

**Abstract**: Face recognition (FR) technology plays a crucial role in various applications, but its vulnerability to adversarial attacks poses significant security concerns. Existing research primarily focuses on transferability to different FR models, overlooking the direct transferability to victim's face images, which is a practical threat in real-world scenarios. In this study, we propose a novel adversarial attack method that considers both the transferability to the FR model and the victim's face image, called NeRFTAP. Leveraging NeRF-based 3D-GAN, we generate new view face images for the source and target subjects to enhance transferability of adversarial patches. We introduce a style consistency loss to ensure the visual similarity between the adversarial UV map and the target UV map under a 0-1 mask, enhancing the effectiveness and naturalness of the generated adversarial face images. Extensive experiments and evaluations on various FR models demonstrate the superiority of our approach over existing attack techniques. Our work provides valuable insights for enhancing the robustness of FR systems in practical adversarial settings.



## **48. Content-based Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2305.10665v2) [paper-pdf](http://arxiv.org/pdf/2305.10665v2)

**Authors**: Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4% and 16.8-48.0% in normally trained models and defense methods, respectively.



## **49. Advancing Attack-Resilient Scheduling of Integrated Energy Systems with Demand Response via Deep Reinforcement Learning**

eess.SY

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17941v1) [paper-pdf](http://arxiv.org/pdf/2311.17941v1)

**Authors**: Yang Li, Wenjie Ma, Yuanzheng Li, Sen Li, Zhe Chen

**Abstract**: Optimally scheduling multi-energy flow is an effective method to utilize renewable energy sources (RES) and improve the stability and economy of integrated energy systems (IES). However, the stable demand-supply of IES faces challenges from uncertainties that arise from RES and loads, as well as the increasing impact of cyber-attacks with advanced information and communication technologies adoption. To address these challenges, this paper proposes an innovative model-free resilience scheduling method based on state-adversarial deep reinforcement learning (DRL) for integrated demand response (IDR)-enabled IES. The proposed method designs an IDR program to explore the interaction ability of electricity-gas-heat flexible loads. Additionally, a state-adversarial Markov decision process (SA-MDP) model characterizes the energy scheduling problem of IES under cyber-attack. The state-adversarial soft actor-critic (SA-SAC) algorithm is proposed to mitigate the impact of cyber-attacks on the scheduling strategy. Simulation results demonstrate that our method is capable of adequately addressing the uncertainties resulting from RES and loads, mitigating the impact of cyber-attacks on the scheduling strategy, and ensuring a stable demand supply for various energy sources. Moreover, the proposed method demonstrates resilience against cyber-attacks. Compared to the original soft actor-critic (SAC) algorithm, it achieves a 10\% improvement in economic performance under cyber-attack scenarios.



## **50. Scalable Extraction of Training Data from (Production) Language Models**

cs.LG

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17035v1) [paper-pdf](http://arxiv.org/pdf/2311.17035v1)

**Authors**: Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, Katherine Lee

**Abstract**: This paper studies extractable memorization: training data that an adversary can efficiently extract by querying a machine learning model without prior knowledge of the training dataset. We show an adversary can extract gigabytes of training data from open-source language models like Pythia or GPT-Neo, semi-open models like LLaMA or Falcon, and closed models like ChatGPT. Existing techniques from the literature suffice to attack unaligned models; in order to attack the aligned ChatGPT, we develop a new divergence attack that causes the model to diverge from its chatbot-style generations and emit training data at a rate 150x higher than when behaving properly. Our methods show practical attacks can recover far more data than previously thought, and reveal that current alignment techniques do not eliminate memorization.



