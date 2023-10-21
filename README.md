# Latest Adversarial Attack Papers
**update at 2023-10-21 11:22:52**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. OODRobustBench: benchmarking and analyzing adversarial robustness under distribution shift**

cs.LG

in submission

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12793v1) [paper-pdf](http://arxiv.org/pdf/2310.12793v1)

**Authors**: Lin Li, Yifei Wang, Chawin Sitawarin, Michael Spratling

**Abstract**: Existing works have made great progress in improving adversarial robustness, but typically test their method only on data from the same distribution as the training data, i.e. in-distribution (ID) testing. As a result, it is unclear how such robustness generalizes under input distribution shifts, i.e. out-of-distribution (OOD) testing. This is a concerning omission as such distribution shifts are unavoidable when methods are deployed in the wild. To address this issue we propose a benchmark named OODRobustBench to comprehensively assess OOD adversarial robustness using 23 dataset-wise shifts (i.e. naturalistic shifts in input distribution) and 6 threat-wise shifts (i.e., unforeseen adversarial threat models). OODRobustBench is used to assess 706 robust models using 60.7K adversarial evaluations. This large-scale analysis shows that: 1) adversarial robustness suffers from a severe OOD generalization issue; 2) ID robustness correlates strongly with OOD robustness, in a positive linear way, under many distribution shifts. The latter enables the prediction of OOD robustness from ID robustness. Based on this, we are able to predict the upper limit of OOD robustness for existing robust training schemes. The results suggest that achieving OOD robustness requires designing novel methods beyond the conventional ones. Last, we discover that extra data, data augmentation, advanced model architectures and particular regularization approaches can improve OOD robustness. Noticeably, the discovered training schemes, compared to the baseline, exhibit dramatically higher robustness under threat shift while keeping high ID robustness, demonstrating new promising solutions for robustness against both multi-attack and unforeseen attacks.



## **2. TorKameleon: Improving Tor's Censorship Resistance with K-anonymization and Media-based Covert Channels**

cs.CR

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2303.17544v3) [paper-pdf](http://arxiv.org/pdf/2303.17544v3)

**Authors**: Afonso Vilalonga, João S. Resende, Henrique Domingos

**Abstract**: Anonymity networks like Tor significantly enhance online privacy but are vulnerable to correlation attacks by state-level adversaries. While covert channels encapsulated in media protocols, particularly WebRTC-based encapsulation, have demonstrated effectiveness against passive traffic correlation attacks, their resilience against active correlation attacks remains unexplored, and their compatibility with Tor has been limited. This paper introduces TorKameleon, a censorship evasion solution designed to protect Tor users from both passive and active correlation attacks. TorKameleon employs K-anonymization techniques to fragment and reroute traffic through multiple TorKameleon proxies, while also utilizing covert WebRTC-based channels or TLS tunnels to encapsulate user traffic.



## **3. WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor Attacks Against Deep Neural Networks**

cs.CV

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.11595v2) [paper-pdf](http://arxiv.org/pdf/2310.11595v2)

**Authors**: Jun Xia, Zhihao Yue, Yingbo Zhou, Zhiwei Ling, Xian Wei, Mingsong Chen

**Abstract**: Due to the popularity of Artificial Intelligence (AI) technology, numerous backdoor attacks are designed by adversaries to mislead deep neural network predictions by manipulating training samples and training processes. Although backdoor attacks are effective in various real scenarios, they still suffer from the problems of both low fidelity of poisoned samples and non-negligible transfer in latent space, which make them easily detectable by existing backdoor detection algorithms. To overcome the weakness, this paper proposes a novel frequency-based backdoor attack method named WaveAttack, which obtains image high-frequency features through Discrete Wavelet Transform (DWT) to generate backdoor triggers. Furthermore, we introduce an asymmetric frequency obfuscation method, which can add an adaptive residual in the training and inference stage to improve the impact of triggers and further enhance the effectiveness of WaveAttack. Comprehensive experimental results show that WaveAttack not only achieves higher stealthiness and effectiveness, but also outperforms state-of-the-art (SOTA) backdoor attack methods in the fidelity of images by up to 28.27\% improvement in PSNR, 1.61\% improvement in SSIM, and 70.59\% reduction in IS.



## **4. Learn from the Past: A Proxy based Adversarial Defense Framework to Boost Robustness**

cs.LG

16 Pages

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12713v1) [paper-pdf](http://arxiv.org/pdf/2310.12713v1)

**Authors**: Yaohua Liu, Jiaxin Gao, Zhu Liu, Xianghao Jiao, Xin Fan, Risheng Liu

**Abstract**: In light of the vulnerability of deep learning models to adversarial samples and the ensuing security issues, a range of methods, including Adversarial Training (AT) as a prominent representative, aimed at enhancing model robustness against various adversarial attacks, have seen rapid development. However, existing methods essentially assist the current state of target model to defend against parameter-oriented adversarial attacks with explicit or implicit computation burdens, which also suffers from unstable convergence behavior due to inconsistency of optimization trajectories. Diverging from previous work, this paper reconsiders the update rule of target model and corresponding deficiency to defend based on its current state. By introducing the historical state of the target model as a proxy, which is endowed with much prior information for defense, we formulate a two-stage update rule, resulting in a general adversarial defense framework, which we refer to as `LAST' ({\bf L}earn from the P{\bf ast}). Besides, we devise a Self Distillation (SD) based defense objective to constrain the update process of the proxy model without the introduction of larger teacher models. Experimentally, we demonstrate consistent and significant performance enhancements by refining a series of single-step and multi-step AT methods (e.g., up to $\bf 9.2\%$ and $\bf 20.5\%$ improvement of Robust Accuracy (RA) on CIFAR10 and CIFAR100 datasets, respectively) across various datasets, backbones and attack modalities, and validate its ability to enhance training stability and ameliorate catastrophic overfitting issues meanwhile.



## **5. Generating Robust Adversarial Examples against Online Social Networks (OSNs)**

cs.MM

26 pages, 9 figures

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12708v1) [paper-pdf](http://arxiv.org/pdf/2310.12708v1)

**Authors**: Jun Liu, Jiantao Zhou, Haiwei Wu, Weiwei Sun, Jinyu Tian

**Abstract**: Online Social Networks (OSNs) have blossomed into prevailing transmission channels for images in the modern era. Adversarial examples (AEs) deliberately designed to mislead deep neural networks (DNNs) are found to be fragile against the inevitable lossy operations conducted by OSNs. As a result, the AEs would lose their attack capabilities after being transmitted over OSNs. In this work, we aim to design a new framework for generating robust AEs that can survive the OSN transmission; namely, the AEs before and after the OSN transmission both possess strong attack capabilities. To this end, we first propose a differentiable network termed SImulated OSN (SIO) to simulate the various operations conducted by an OSN. Specifically, the SIO network consists of two modules: 1) a differentiable JPEG layer for approximating the ubiquitous JPEG compression and 2) an encoder-decoder subnetwork for mimicking the remaining operations. Based upon the SIO network, we then formulate an optimization framework to generate robust AEs by enforcing model outputs with and without passing through the SIO to be both misled. Extensive experiments conducted over Facebook, WeChat and QQ demonstrate that our attack methods produce more robust AEs than existing approaches, especially under small distortion constraints; the performance gain in terms of Attack Success Rate (ASR) could be more than 60%. Furthermore, we build a public dataset containing more than 10,000 pairs of AEs processed by Facebook, WeChat or QQ, facilitating future research in the robust AEs generation. The dataset and code are available at https://github.com/csjunjun/RobustOSNAttack.git.



## **6. Blades: A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning**

cs.CR

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2206.05359v3) [paper-pdf](http://arxiv.org/pdf/2206.05359v3)

**Authors**: Shenghui Li, Edith Ngai, Fanghua Ye, Li Ju, Tianru Zhang, Thiemo Voigt

**Abstract**: Federated learning (FL) facilitates distributed training across clients, safeguarding the privacy of their data. The inherent distributed structure of FL introduces vulnerabilities, especially from adversarial (Byzantine) clients aiming to skew local updates to their advantage. Despite the plethora of research focusing on Byzantine-resilient FL, the academic community has yet to establish a comprehensive benchmark suite, pivotal for impartial assessment and comparison of different techniques.   This paper investigates existing techniques in Byzantine-resilient FL and introduces an open-source benchmark suite for convenient and fair performance comparisons. Our investigation begins with a systematic study of Byzantine attack and defense strategies. Subsequently, we present \ours, a scalable, extensible, and easily configurable benchmark suite that supports researchers and developers in efficiently implementing and validating novel strategies against baseline algorithms in Byzantine-resilient FL. The design of \ours incorporates key characteristics derived from our systematic study, encompassing the attacker's capabilities and knowledge, defense strategy categories, and factors influencing robustness. Blades contains built-in implementations of representative attack and defense strategies and offers user-friendly interfaces for seamlessly integrating new ideas.



## **7. Automatic Hallucination Assessment for Aligned Large Language Models via Transferable Adversarial Attacks**

cs.CL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12516v1) [paper-pdf](http://arxiv.org/pdf/2310.12516v1)

**Authors**: Xiaodong Yu, Hao Cheng, Xiaodong Liu, Dan Roth, Jianfeng Gao

**Abstract**: Although remarkable progress has been achieved in preventing large language model (LLM) hallucinations using instruction tuning and retrieval augmentation, it remains challenging to measure the reliability of LLMs using human-crafted evaluation data which is not available for many tasks and domains and could suffer from data leakage. Inspired by adversarial machine learning, this paper aims to develop a method of automatically generating evaluation data by appropriately modifying existing data on which LLMs behave faithfully. Specifically, this paper presents AutoDebug, an LLM-based framework to use prompting chaining to generate transferable adversarial attacks in the form of question-answering examples. We seek to understand the extent to which these examples trigger the hallucination behaviors of LLMs.   We implement AutoDebug using ChatGPT and evaluate the resulting two variants of a popular open-domain question-answering dataset, Natural Questions (NQ), on a collection of open-source and proprietary LLMs under various prompting settings. Our generated evaluation data is human-readable and, as we show, humans can answer these modified questions well. Nevertheless, we observe pronounced accuracy drops across multiple LLMs including GPT-4. Our experimental results show that LLMs are likely to hallucinate in two categories of question-answering scenarios where (1) there are conflicts between knowledge given in the prompt and their parametric knowledge, or (2) the knowledge expressed in the prompt is complex. Finally, we find that the adversarial examples generated by our method are transferable across all considered LLMs. The examples generated by a small model can be used to debug a much larger model, making our approach cost-effective.



## **8. Red Teaming Language Model Detectors with Language Models**

cs.CL

Preprint. Accepted by TACL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2305.19713v2) [paper-pdf](http://arxiv.org/pdf/2305.19713v2)

**Authors**: Zhouxing Shi, Yihan Wang, Fan Yin, Xiangning Chen, Kai-Wei Chang, Cho-Jui Hsieh

**Abstract**: The prevalence and strong capability of large language models (LLMs) present significant safety and ethical risks if exploited by malicious users. To prevent the potentially deceptive usage of LLMs, recent works have proposed algorithms to detect LLM-generated text and protect LLMs. In this paper, we investigate the robustness and reliability of these LLM detectors under adversarial attacks. We study two types of attack strategies: 1) replacing certain words in an LLM's output with their synonyms given the context; 2) automatically searching for an instructional prompt to alter the writing style of the generation. In both strategies, we leverage an auxiliary LLM to generate the word replacements or the instructional prompt. Different from previous works, we consider a challenging setting where the auxiliary LLM can also be protected by a detector. Experiments reveal that our attacks effectively compromise the performance of all detectors in the study with plausible generations, underscoring the urgent need to improve the robustness of LLM-generated text detection systems.



## **9. Unraveling the Connections between Privacy and Certified Robustness in Federated Learning Against Poisoning Attacks**

cs.CR

ACM CCS 2023

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2209.04030v2) [paper-pdf](http://arxiv.org/pdf/2209.04030v2)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Qinbin Li, Sanmi Koyejo, Bo Li

**Abstract**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As local training data comes from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is usually trained in a differentially private way (DPFL). Thus, in this paper, we ask: What are the underlying connections between differential privacy and certified robustness in FL against poisoning attacks? Can we leverage the innate privacy property of DPFL to provide certified robustness for FL? Can we further improve the privacy of FL to improve such robustness certification? We first investigate both user-level and instance-level privacy of FL and provide formal privacy analysis to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack inefficacy for DPFL on both user and instance levels. Theoretically, we provide the certified robustness of DPFL based on both criteria given a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of poisoning attacks on different datasets. We find that increasing the level of privacy protection in DPFL results in stronger certified attack inefficacy; however, it does not necessarily lead to a stronger certified prediction. Thus, achieving the optimal certified prediction requires a proper balance between privacy and utility loss.



## **10. CAT: Closed-loop Adversarial Training for Safe End-to-End Driving**

cs.LG

7th Conference on Robot Learning (CoRL 2023)

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12432v1) [paper-pdf](http://arxiv.org/pdf/2310.12432v1)

**Authors**: Linrui Zhang, Zhenghao Peng, Quanyi Li, Bolei Zhou

**Abstract**: Driving safety is a top priority for autonomous vehicles. Orthogonal to prior work handling accident-prone traffic events by algorithm designs at the policy level, we investigate a Closed-loop Adversarial Training (CAT) framework for safe end-to-end driving in this paper through the lens of environment augmentation. CAT aims to continuously improve the safety of driving agents by training the agent on safety-critical scenarios that are dynamically generated over time. A novel resampling technique is developed to turn log-replay real-world driving scenarios into safety-critical ones via probabilistic factorization, where the adversarial traffic generation is modeled as the multiplication of standard motion prediction sub-problems. Consequently, CAT can launch more efficient physical attacks compared to existing safety-critical scenario generation methods and yields a significantly less computational cost in the iterative learning pipeline. We incorporate CAT into the MetaDrive simulator and validate our approach on hundreds of driving scenarios imported from real-world driving datasets. Experimental results demonstrate that CAT can effectively generate adversarial scenarios countering the agent being trained. After training, the agent can achieve superior driving safety in both log-replay and safety-critical traffic scenarios on the held-out test set. Code and data are available at https://metadriverse.github.io/cat.



## **11. Segment Anything Meets Universal Adversarial Perturbation**

cs.CV

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12431v1) [paper-pdf](http://arxiv.org/pdf/2310.12431v1)

**Authors**: Dongshen Han, Sheng Zheng, Chaoning Zhang

**Abstract**: As Segment Anything Model (SAM) becomes a popular foundation model in computer vision, its adversarial robustness has become a concern that cannot be ignored. This works investigates whether it is possible to attack SAM with image-agnostic Universal Adversarial Perturbation (UAP). In other words, we seek a single perturbation that can fool the SAM to predict invalid masks for most (if not all) images. We demonstrate convetional image-centric attack framework is effective for image-independent attacks but fails for universal adversarial attack. To this end, we propose a novel perturbation-centric framework that results in a UAP generation method based on self-supervised contrastive learning (CL), where the UAP is set to the anchor sample and the positive sample is augmented from the UAP. The representations of negative samples are obtained from the image encoder in advance and saved in a memory bank. The effectiveness of our proposed CL-based UAP generation method is validated by both quantitative and qualitative results. On top of the ablation study to understand various components in our proposed method, we shed light on the roles of positive and negative samples in making the generated UAP effective for attacking SAM.



## **12. One-Bit Byzantine-Tolerant Distributed Learning via Over-the-Air Computation**

eess.SP

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.11998v2) [paper-pdf](http://arxiv.org/pdf/2310.11998v2)

**Authors**: Yuhan Yang, Youlong Wu, Yuning Jiang, Yuanming Shi

**Abstract**: Distributed learning has become a promising computational parallelism paradigm that enables a wide scope of intelligent applications from the Internet of Things (IoT) to autonomous driving and the healthcare industry. This paper studies distributed learning in wireless data center networks, which contain a central edge server and multiple edge workers to collaboratively train a shared global model and benefit from parallel computing. However, the distributed nature causes the vulnerability of the learning process to faults and adversarial attacks from Byzantine edge workers, as well as the severe communication and computation overhead induced by the periodical information exchange process. To achieve fast and reliable model aggregation in the presence of Byzantine attacks, we develop a signed stochastic gradient descent (SignSGD)-based Hierarchical Vote framework via over-the-air computation (AirComp), where one voting process is performed locally at the wireless edge by taking advantage of Bernoulli coding while the other is operated over-the-air at the central edge server by utilizing the waveform superposition property of the multiple-access channels. We comprehensively analyze the proposed framework on the impacts including Byzantine attacks and the wireless environment (channel fading and receiver noise), followed by characterizing the convergence behavior under non-convex settings. Simulation results validate our theoretical achievements and demonstrate the robustness of our proposed framework in the presence of Byzantine attacks and receiver noise.



## **13. REVAMP: Automated Simulations of Adversarial Attacks on Arbitrary Objects in Realistic Scenes**

cs.LG

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.12243v1) [paper-pdf](http://arxiv.org/pdf/2310.12243v1)

**Authors**: Matthew Hull, Zijie J. Wang, Duen Horng Chau

**Abstract**: Deep Learning models, such as those used in an autonomous vehicle are vulnerable to adversarial attacks where an attacker could place an adversarial object in the environment, leading to mis-classification. Generating these adversarial objects in the digital space has been extensively studied, however successfully transferring these attacks from the digital realm to the physical realm has proven challenging when controlling for real-world environmental factors. In response to these limitations, we introduce REVAMP, an easy-to-use Python library that is the first-of-its-kind tool for creating attack scenarios with arbitrary objects and simulating realistic environmental factors, lighting, reflection, and refraction. REVAMP enables researchers and practitioners to swiftly explore various scenarios within the digital realm by offering a wide range of configurable options for designing experiments and using differentiable rendering to reproduce physically plausible adversarial objects. We will demonstrate and invite the audience to try REVAMP to produce an adversarial texture on a chosen object while having control over various scene parameters. The audience will choose a scene, an object to attack, the desired attack class, and the number of camera positions to use. Then, in real time, we show how this altered texture causes the chosen object to be mis-classified, showcasing the potential of REVAMP in real-world scenarios. REVAMP is open-source and available at https://github.com/poloclub/revamp.



## **14. A Black-Box Attack on Code Models via Representation Nearest Neighbor Search**

cs.CR

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2305.05896v3) [paper-pdf](http://arxiv.org/pdf/2305.05896v3)

**Authors**: Jie Zhang, Wei Ma, Qiang Hu, Shangqing Liu, Xiaofei Xie, Yves Le Traon, Yang Liu

**Abstract**: Existing methods for generating adversarial code examples face several challenges: limted availability of substitute variables, high verification costs for these substitutes, and the creation of adversarial samples with noticeable perturbations. To address these concerns, our proposed approach, RNNS, uses a search seed based on historical attacks to find potential adversarial substitutes. Rather than directly using the discrete substitutes, they are mapped to a continuous vector space using a pre-trained variable name encoder. Based on the vector representation, RNNS predicts and selects better substitutes for attacks. We evaluated the performance of RNNS across six coding tasks encompassing three programming languages: Java, Python, and C. We employed three pre-trained code models (CodeBERT, GraphCodeBERT, and CodeT5) that resulted in a cumulative of 18 victim models. The results demonstrate that RNNS outperforms baselines in terms of ASR and QT. Furthermore, the perturbation of adversarial examples introduced by RNNS is smaller compared to the baselines in terms of the number of replaced variables and the change in variable length. Lastly, our experiments indicate that RNNS is efficient in attacking defended models and can be employed for adversarial training.



## **15. Black-Box Training Data Identification in GANs via Detector Networks**

cs.LG

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.12063v1) [paper-pdf](http://arxiv.org/pdf/2310.12063v1)

**Authors**: Lukman Olagoke, Salil Vadhan, Seth Neel

**Abstract**: Since their inception Generative Adversarial Networks (GANs) have been popular generative models across images, audio, video, and tabular data. In this paper we study whether given access to a trained GAN, as well as fresh samples from the underlying distribution, if it is possible for an attacker to efficiently identify if a given point is a member of the GAN's training data. This is of interest for both reasons related to copyright, where a user may want to determine if their copyrighted data has been used to train a GAN, and in the study of data privacy, where the ability to detect training set membership is known as a membership inference attack. Unlike the majority of prior work this paper investigates the privacy implications of using GANs in black-box settings, where the attack only has access to samples from the generator, rather than access to the discriminator as well. We introduce a suite of membership inference attacks against GANs in the black-box setting and evaluate our attacks on image GANs trained on the CIFAR10 dataset and tabular GANs trained on genomic data. Our most successful attack, called The Detector, involve training a second network to score samples based on their likelihood of being generated by the GAN, as opposed to a fresh sample from the distribution. We prove under a simple model of the generator that the detector is an approximately optimal membership inference attack. Across a wide range of tabular and image datasets, attacks, and GAN architectures, we find that adversaries can orchestrate non-trivial privacy attacks when provided with access to samples from the generator. At the same time, the attack success achievable against GANs still appears to be lower compared to other generative and discriminative models; this leaves the intriguing open question of whether GANs are in fact more private, or if it is a matter of developing stronger attacks.



## **16. Exploring Decision-based Black-box Attacks on Face Forgery Detection**

cs.CV

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.12017v1) [paper-pdf](http://arxiv.org/pdf/2310.12017v1)

**Authors**: Zhaoyu Chen, Bo Li, Kaixun Jiang, Shuang Wu, Shouhong Ding, Wenqiang Zhang

**Abstract**: Face forgery generation technologies generate vivid faces, which have raised public concerns about security and privacy. Many intelligent systems, such as electronic payment and identity verification, rely on face forgery detection. Although face forgery detection has successfully distinguished fake faces, recent studies have demonstrated that face forgery detectors are very vulnerable to adversarial examples. Meanwhile, existing attacks rely on network architectures or training datasets instead of the predicted labels, which leads to a gap in attacking deployed applications. To narrow this gap, we first explore the decision-based attacks on face forgery detection. However, applying existing decision-based attacks directly suffers from perturbation initialization failure and low image quality. First, we propose cross-task perturbation to handle initialization failures by utilizing the high correlation of face features on different tasks. Then, inspired by using frequency cues by face forgery detection, we propose the frequency decision-based attack. We add perturbations in the frequency domain and then constrain the visual quality in the spatial domain. Finally, extensive experiments demonstrate that our method achieves state-of-the-art attack performance on FaceForensics++, CelebDF, and industrial APIs, with high query efficiency and guaranteed image quality. Further, the fake faces by our method can pass face forgery detection and face recognition, which exposes the security problems of face forgery detectors.



## **17. Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2309.13609v2) [paper-pdf](http://arxiv.org/pdf/2309.13609v2)

**Authors**: Ao-Xiang Zhang, Yu Ran, Weixuan Tang, Yuan-Gen Wang

**Abstract**: No-Reference Video Quality Assessment (NR-VQA) plays an essential role in improving the viewing experience of end-users. Driven by deep learning, recent NR-VQA models based on Convolutional Neural Networks (CNNs) and Transformers have achieved outstanding performance. To build a reliable and practical assessment system, it is of great necessity to evaluate their robustness. However, such issue has received little attention in the academic community. In this paper, we make the first attempt to evaluate the robustness of NR-VQA models against adversarial attacks, and propose a patch-based random search method for black-box attack. Specifically, considering both the attack effect on quality score and the visual quality of adversarial video, the attack problem is formulated as misleading the estimated quality score under the constraint of just-noticeable difference (JND). Built upon such formulation, a novel loss function called Score-Reversed Boundary Loss is designed to push the adversarial video's estimated quality score far away from its ground-truth score towards a specific boundary, and the JND constraint is modeled as a strict $L_2$ and $L_\infty$ norm restriction. By this means, both white-box and black-box attacks can be launched in an effective and imperceptible manner. The source code is available at https://github.com/GZHU-DVL/AttackVQA.



## **18. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2306.04528v4) [paper-pdf](http://arxiv.org/pdf/2306.04528v4)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. Code is available at: https://github.com/microsoft/promptbench.



## **19. Quantifying Privacy Risks of Prompts in Visual Prompt Learning**

cs.CR

To appear in the 33rd USENIX Security Symposium, August 14-16, 2024

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11970v1) [paper-pdf](http://arxiv.org/pdf/2310.11970v1)

**Authors**: Yixin Wu, Rui Wen, Michael Backes, Pascal Berrang, Mathias Humbert, Yun Shen, Yang Zhang

**Abstract**: Large-scale pre-trained models are increasingly adapted to downstream tasks through a new paradigm called prompt learning. In contrast to fine-tuning, prompt learning does not update the pre-trained model's parameters. Instead, it only learns an input perturbation, namely prompt, to be added to the downstream task data for predictions. Given the fast development of prompt learning, a well-generalized prompt inevitably becomes a valuable asset as significant effort and proprietary data are used to create it. This naturally raises the question of whether a prompt may leak the proprietary information of its training data. In this paper, we perform the first comprehensive privacy assessment of prompts learned by visual prompt learning through the lens of property inference and membership inference attacks. Our empirical evaluation shows that the prompts are vulnerable to both attacks. We also demonstrate that the adversary can mount a successful property inference attack with limited cost. Moreover, we show that membership inference attacks against prompts can be successful with relaxed adversarial assumptions. We further make some initial investigations on the defenses and observe that our method can mitigate the membership inference attacks with a decent utility-defense trade-off but fails to defend against property inference attacks. We hope our results can shed light on the privacy risks of the popular prompt learning paradigm. To facilitate the research in this direction, we will share our code and models with the community.



## **20. IMAP: Intrinsically Motivated Adversarial Policy**

cs.LG

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2305.02605v2) [paper-pdf](http://arxiv.org/pdf/2305.02605v2)

**Authors**: Xiang Zheng, Xingjun Ma, Shengjie Wang, Xinyu Wang, Chao Shen, Cong Wang

**Abstract**: Reinforcement learning agents are susceptible to evasion attacks during deployment. In single-agent environments, these attacks can occur through imperceptible perturbations injected into the inputs of the victim policy network. In multi-agent environments, an attacker can manipulate an adversarial opponent to influence the victim policy's observations indirectly. While adversarial policies offer a promising technique to craft such attacks, current methods are either sample-inefficient due to poor exploration strategies or require extra surrogate model training under the black-box assumption. To address these challenges, in this paper, we propose Intrinsically Motivated Adversarial Policy (IMAP) for efficient black-box adversarial policy learning in both single- and multi-agent environments. We formulate four types of adversarial intrinsic regularizers -- maximizing the adversarial state coverage, policy coverage, risk, or divergence -- to discover potential vulnerabilities of the victim policy in a principled way. We also present a novel Bias-Reduction (BR) method to boost IMAP further. Our experiments validate the effectiveness of the four types of adversarial intrinsic regularizers and BR in enhancing black-box adversarial policy learning across a variety of environments. Our IMAP successfully evades two types of defense methods, adversarial training and robust regularizer, decreasing the performance of the state-of-the-art robust WocaR-PPO agents by 34%-54% across four single-agent tasks. IMAP also achieves a state-of-the-art attacking success rate of 83.91% in the multi-agent game YouShallNotPass.



## **21. Malicious Agent Detection for Robust Multi-Agent Collaborative Perception**

cs.CR

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11901v1) [paper-pdf](http://arxiv.org/pdf/2310.11901v1)

**Authors**: Yangheng Zhao, Zhen Xiang, Sheng Yin, Xianghe Pang, Siheng Chen, Yanfeng Wang

**Abstract**: Recently, multi-agent collaborative (MAC) perception has been proposed and outperformed the traditional single-agent perception in many applications, such as autonomous driving. However, MAC perception is more vulnerable to adversarial attacks than single-agent perception due to the information exchange. The attacker can easily degrade the performance of a victim agent by sending harmful information from a malicious agent nearby. In this paper, we extend adversarial attacks to an important perception task -- MAC object detection, where generic defenses such as adversarial training are no longer effective against these attacks. More importantly, we propose Malicious Agent Detection (MADE), a reactive defense specific to MAC perception that can be deployed by each agent to accurately detect and then remove any potential malicious agent in its local collaboration network. In particular, MADE inspects each agent in the network independently using a semi-supervised anomaly detector based on a double-hypothesis test with the Benjamini-Hochberg procedure to control the false positive rate of the inference. For the two hypothesis tests, we propose a match loss statistic and a collaborative reconstruction loss statistic, respectively, both based on the consistency between the agent to be inspected and the ego agent where our detector is deployed. We conduct comprehensive evaluations on a benchmark 3D dataset V2X-sim and a real-road dataset DAIR-V2X and show that with the protection of MADE, the drops in the average precision compared with the best-case "oracle" defender against our attack are merely 1.28% and 0.34%, respectively, much lower than 8.92% and 10.00% for adversarial training, respectively.



## **22. IRAD: Implicit Representation-driven Image Resampling against Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11890v1) [paper-pdf](http://arxiv.org/pdf/2310.11890v1)

**Authors**: Yue Cao, Tianlin Li, Xiaofeng Cao, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: We introduce a novel approach to counter adversarial attacks, namely, image resampling. Image resampling transforms a discrete image into a new one, simulating the process of scene recapturing or rerendering as specified by a geometrical transformation. The underlying rationale behind our idea is that image resampling can alleviate the influence of adversarial perturbations while preserving essential semantic information, thereby conferring an inherent advantage in defending against adversarial attacks. To validate this concept, we present a comprehensive study on leveraging image resampling to defend against adversarial attacks. We have developed basic resampling methods that employ interpolation strategies and coordinate shifting magnitudes. Our analysis reveals that these basic methods can partially mitigate adversarial attacks. However, they come with apparent limitations: the accuracy of clean images noticeably decreases, while the improvement in accuracy on adversarial examples is not substantial. We propose implicit representation-driven image resampling (IRAD) to overcome these limitations. First, we construct an implicit continuous representation that enables us to represent any input image within a continuous coordinate space. Second, we introduce SampleNet, which automatically generates pixel-wise shifts for resampling in response to different inputs. Furthermore, we can extend our approach to the state-of-the-art diffusion-based method, accelerating it with fewer time steps while preserving its defense capability. Extensive experiments demonstrate that our method significantly enhances the adversarial robustness of diverse deep models against various attacks while maintaining high accuracy on clean images.



## **23. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

cs.CV

Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11868v1) [paper-pdf](http://arxiv.org/pdf/2310.11868v1)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of complex and diverse images. However, these models also introduce potential safety hazards, such as the production of harmful content and infringement of data copyrights. Although there have been efforts to create safety-driven unlearning methods to counteract these challenges, doubts remain about their capabilities. To bridge this uncertainty, we propose an evaluation framework built upon adversarial attacks (also referred to as adversarial prompts), in order to discern the trustworthiness of these safety-driven unlearned DMs. Specifically, our research explores the (worst-case) robustness of unlearned DMs in eradicating unwanted concepts, styles, and objects, assessed by the generation of adversarial prompts. We develop a novel adversarial learning approach called UnlearnDiff that leverages the inherent classification capabilities of DMs to streamline the generation of adversarial prompts, making it as simple for DMs as it is for image classification attacks. This technique streamlines the creation of adversarial prompts, making the process as intuitive for generative modeling as it is for image classification assaults. Through comprehensive benchmarking, we assess the unlearning robustness of five prevalent unlearned DMs across multiple tasks. Our results underscore the effectiveness and efficiency of UnlearnDiff when compared to state-of-the-art adversarial prompting methods. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: This paper contains model outputs that may be offensive in nature.



## **24. Revisiting Transferable Adversarial Image Examples: Attack Categorization, Evaluation Guidelines, and New Insights**

cs.CR

Code is available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11850v1) [paper-pdf](http://arxiv.org/pdf/2310.11850v1)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes, Qi Li, Chao Shen

**Abstract**: Transferable adversarial examples raise critical security concerns in real-world, black-box attack scenarios. However, in this work, we identify two main problems in common evaluation practices: (1) For attack transferability, lack of systematic, one-to-one attack comparison and fair hyperparameter settings. (2) For attack stealthiness, simply no comparisons. To address these problems, we establish new evaluation guidelines by (1) proposing a novel attack categorization strategy and conducting systematic and fair intra-category analyses on transferability, and (2) considering diverse imperceptibility metrics and finer-grained stealthiness characteristics from the perspective of attack traceback. To this end, we provide the first large-scale evaluation of transferable adversarial examples on ImageNet, involving 23 representative attacks against 9 representative defenses. Our evaluation leads to a number of new insights, including consensus-challenging ones: (1) Under a fair attack hyperparameter setting, one early attack method, DI, actually outperforms all the follow-up methods. (2) A state-of-the-art defense, DiffPure, actually gives a false sense of (white-box) security since it is indeed largely bypassed by our (black-box) transferable attacks. (3) Even when all attacks are bounded by the same $L_p$ norm, they lead to dramatically different stealthiness performance, which negatively correlates with their transferability performance. Overall, our work demonstrates that existing problematic evaluations have indeed caused misleading conclusions and missing points, and as a result, hindered the assessment of the actual progress in this field.



## **25. Adversarial Training for Physics-Informed Neural Networks**

cs.LG

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11789v1) [paper-pdf](http://arxiv.org/pdf/2310.11789v1)

**Authors**: Yao Li, Shengzhu Shi, Zhichang Guo, Boying Wu

**Abstract**: Physics-informed neural networks have shown great promise in solving partial differential equations. However, due to insufficient robustness, vanilla PINNs often face challenges when solving complex PDEs, especially those involving multi-scale behaviors or solutions with sharp or oscillatory characteristics. To address these issues, based on the projected gradient descent adversarial attack, we proposed an adversarial training strategy for PINNs termed by AT-PINNs. AT-PINNs enhance the robustness of PINNs by fine-tuning the model with adversarial samples, which can accurately identify model failure locations and drive the model to focus on those regions during training. AT-PINNs can also perform inference with temporal causality by selecting the initial collocation points around temporal initial values. We implement AT-PINNs to the elliptic equation with multi-scale coefficients, Poisson equation with multi-peak solutions, Burgers equation with sharp solutions and the Allen-Cahn equation. The results demonstrate that AT-PINNs can effectively locate and reduce failure regions. Moreover, AT-PINNs are suitable for solving complex PDEs, since locating failure regions through adversarial attacks is independent of the size of failure regions or the complexity of the distribution.



## **26. Evading Watermark based Detection of AI-Generated Content**

cs.LG

To appear in ACM Conference on Computer and Communications Security  (CCS), 2023

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2305.03807v4) [paper-pdf](http://arxiv.org/pdf/2305.03807v4)

**Authors**: Zhengyuan Jiang, Jinghuai Zhang, Neil Zhenqiang Gong

**Abstract**: A generative AI model can generate extremely realistic-looking content, posing growing challenges to the authenticity of information. To address the challenges, watermark has been leveraged to detect AI-generated content. Specifically, a watermark is embedded into an AI-generated content before it is released. A content is detected as AI-generated if a similar watermark can be decoded from it. In this work, we perform a systematic study on the robustness of such watermark-based AI-generated content detection. We focus on AI-generated images. Our work shows that an attacker can post-process a watermarked image via adding a small, human-imperceptible perturbation to it, such that the post-processed image evades detection while maintaining its visual quality. We show the effectiveness of our attack both theoretically and empirically. Moreover, to evade detection, our adversarial post-processing method adds much smaller perturbations to AI-generated images and thus better maintain their visual quality than existing popular post-processing methods such as JPEG compression, Gaussian blur, and Brightness/Contrast. Our work shows the insufficiency of existing watermark-based detection of AI-generated content, highlighting the urgent needs of new methods. Our code is publicly available: https://github.com/zhengyuan-jiang/WEvade.



## **27. Attacks Meet Interpretability (AmI) Evaluation and Findings**

cs.CR

5 pages, 4 figures

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.08808v2) [paper-pdf](http://arxiv.org/pdf/2310.08808v2)

**Authors**: Qian Ma, Ziping Ye, Shagufta Mehnaz

**Abstract**: To investigate the effectiveness of the model explanation in detecting adversarial examples, we reproduce the results of two papers, Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples and Is AmI (Attacks Meet Interpretability) Robust to Adversarial Examples. And then conduct experiments and case studies to identify the limitations of both works. We find that Attacks Meet Interpretability(AmI) is highly dependent on the selection of hyperparameters. Therefore, with a different hyperparameter choice, AmI is still able to detect Nicholas Carlini's attack. Finally, we propose recommendations for future work on the evaluation of defense techniques such as AmI.



## **28. The Efficacy of Transformer-based Adversarial Attacks in Security Domains**

cs.CR

Accepted to IEEE Military Communications Conference (MILCOM), AI for  Cyber Workshop, 2023

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2310.11597v1) [paper-pdf](http://arxiv.org/pdf/2310.11597v1)

**Authors**: Kunyang Li, Kyle Domico, Jean-Charles Noirot Ferrand, Patrick McDaniel

**Abstract**: Today, the security of many domains rely on the use of Machine Learning to detect threats, identify vulnerabilities, and safeguard systems from attacks. Recently, transformer architectures have improved the state-of-the-art performance on a wide range of tasks such as malware detection and network intrusion detection. But, before abandoning current approaches to transformers, it is crucial to understand their properties and implications on cybersecurity applications. In this paper, we evaluate the robustness of transformers to adversarial samples for system defenders (i.e., resiliency to adversarial perturbations generated on different types of architectures) and their adversarial strength for system attackers (i.e., transferability of adversarial samples generated by transformers to other target models). To that effect, we first fine-tune a set of pre-trained transformer, Convolutional Neural Network (CNN), and hybrid (an ensemble of transformer and CNN) models to solve different downstream image-based tasks. Then, we use an attack algorithm to craft 19,367 adversarial examples on each model for each task. The transferability of these adversarial examples is measured by evaluating each set on other models to determine which models offer more adversarial strength, and consequently, more robustness against these attacks. We find that the adversarial examples crafted on transformers offer the highest transferability rate (i.e., 25.7% higher than the average) onto other models. Similarly, adversarial examples crafted on other models have the lowest rate of transferability (i.e., 56.7% lower than the average) onto transformers. Our work emphasizes the importance of studying transformer architectures for attacking and defending models in security domains, and suggests using them as the primary architecture in transfer attack settings.



## **29. Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning**

cs.LG

8 pages, 6 main pages of text, 4 figures, 2 tables. Made for a  Neurips workshop on backdoor attacks

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2310.11594v1) [paper-pdf](http://arxiv.org/pdf/2310.11594v1)

**Authors**: Taejin Kim, Jiarui Li, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: In today's data-driven landscape, the delicate equilibrium between safeguarding user privacy and unleashing data potential stands as a paramount concern. Federated learning, which enables collaborative model training without necessitating data sharing, has emerged as a privacy-centric solution. This decentralized approach brings forth security challenges, notably poisoning and backdoor attacks where malicious entities inject corrupted data. Our research, initially spurred by test-time evasion attacks, investigates the intersection of adversarial training and backdoor attacks within federated learning, introducing Adversarial Robustness Unhardening (ARU). ARU is employed by a subset of adversaries to intentionally undermine model robustness during decentralized training, rendering models susceptible to a broader range of evasion attacks. We present extensive empirical experiments evaluating ARU's impact on adversarial training and existing robust aggregation defenses against poisoning and backdoor attacks. Our findings inform strategies for enhancing ARU to counter current defensive measures and highlight the limitations of existing defenses, offering insights into bolstering defenses against ARU.



## **30. A Two-Layer Blockchain Sharding Protocol Leveraging Safety and Liveness for Enhanced Performance**

cs.CR

The paper has been accepted to Network and Distributed System  Security (NDSS) Symposium 2024

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2310.11373v1) [paper-pdf](http://arxiv.org/pdf/2310.11373v1)

**Authors**: Yibin Xu, Jingyi Zheng, Boris Düdder, Tijs Slaats, Yongluan Zhou

**Abstract**: Sharding is essential for improving blockchain scalability. Existing protocols overlook diverse adversarial attacks, limiting transaction throughput. This paper presents Reticulum, a groundbreaking sharding protocol addressing this issue, boosting blockchain scalability.   Reticulum employs a two-phase approach, adapting transaction throughput based on runtime adversarial attacks. It comprises "control" and "process" shards in two layers. Process shards contain at least one trustworthy node, while control shards have a majority of trusted nodes. In the first phase, transactions are written to blocks and voted on by nodes in process shards. Unanimously accepted blocks are confirmed. In the second phase, blocks without unanimous acceptance are voted on by control shards. Blocks are accepted if the majority votes in favor, eliminating first-phase opponents and silent voters. Reticulum uses unanimous voting in the first phase, involving fewer nodes, enabling more parallel process shards. Control shards finalize decisions and resolve disputes.   Experiments confirm Reticulum's innovative design, providing high transaction throughput and robustness against various network attacks, outperforming existing sharding protocols for blockchain networks.



## **31. CARSO: Blending Adversarial Training and Purification Improves Adversarial Robustness**

cs.CV

19 pages, 1 figure, 9 tables

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2306.06081v3) [paper-pdf](http://arxiv.org/pdf/2306.06081v3)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this work, we propose a novel adversarial defence mechanism for image classification - CARSO - blending the paradigms of adversarial training and adversarial purification in a mutually-beneficial, robustness-enhancing way. The method builds upon an adversarially-trained classifier, and learns to map its internal representation associated with a potentially perturbed input onto a distribution of tentative clean reconstructions. Multiple samples from such distribution are classified by the adversarially-trained model itself, and an aggregation of its outputs finally constitutes the robust prediction of interest. Experimental evaluation by a well-established benchmark of varied, strong adaptive attacks, across different image datasets and classifier architectures, shows that CARSO is able to defend itself against foreseen and unforeseen threats, including adaptive end-to-end attacks devised for stochastic defences. Paying a tolerable clean accuracy toll, our method improves by a significant margin the state of the art for CIFAR-10 and CIFAR-100 $\ell_\infty$ robust classification accuracy against AutoAttack. Code and pre-trained models are available at https://github.com/emaballarin/CARSO .



## **32. A Comprehensive Study of the Robustness for LiDAR-based 3D Object Detectors against Adversarial Attacks**

cs.CV

30 pages, 14 figures. Accepted by IJCV

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2212.10230v3) [paper-pdf](http://arxiv.org/pdf/2212.10230v3)

**Authors**: Yifan Zhang, Junhui Hou, Yixuan Yuan

**Abstract**: Recent years have witnessed significant advancements in deep learning-based 3D object detection, leading to its widespread adoption in numerous applications. As 3D object detectors become increasingly crucial for security-critical tasks, it is imperative to understand their robustness against adversarial attacks. This paper presents the first comprehensive evaluation and analysis of the robustness of LiDAR-based 3D detectors under adversarial attacks. Specifically, we extend three distinct adversarial attacks to the 3D object detection task, benchmarking the robustness of state-of-the-art LiDAR-based 3D object detectors against attacks on the KITTI and Waymo datasets. We further analyze the relationship between robustness and detector properties. Additionally, we explore the transferability of cross-model, cross-task, and cross-data attacks. Thorough experiments on defensive strategies for 3D detectors are conducted, demonstrating that simple transformations like flipping provide little help in improving robustness when the applied transformation strategy is exposed to attackers. \revise{Finally, we propose balanced adversarial focal training, based on conventional adversarial training, to strike a balance between accuracy and robustness.} Our findings will facilitate investigations into understanding and defending against adversarial attacks on LiDAR-based 3D object detectors, thus advancing the field. The source code is publicly available at \url{https://github.com/Eaphan/Robust3DOD}.



## **33. An adversarially robust data-market for spatial, crowd-sourced data**

cs.DS

13 pages, 7 figures

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2206.06299v3) [paper-pdf](http://arxiv.org/pdf/2206.06299v3)

**Authors**: Aida Manzano Kharman, Christian Jursitzky, Quan Zhou, Pietro Ferraro, Jakub Marecek, Pierre Pinson, Robert Shorten

**Abstract**: We describe an architecture for a decentralised data market for applications in which agents are incentivised to collaborate to crowd-source their data. The architecture is designed to reward data that furthers the market's collective goal, and distributes reward fairly to all those that contribute with their data. We show that the architecture is resilient to Sybil, wormhole, and data poisoning attacks. In order to evaluate the resilience of the architecture, we characterise its breakdown points for various adversarial threat models in an automotive use case.



## **34. Patch of Invisibility: Naturalistic Physical Black-Box Adversarial Attacks on Object Detectors**

cs.CV

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2303.04238v4) [paper-pdf](http://arxiv.org/pdf/2303.04238v4)

**Authors**: Raz Lapid, Eylon Mizrahi, Moshe Sipper

**Abstract**: Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called ``white-box'' attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a direct, black-box, gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. To our knowledge this is the first and only method that performs black-box physical attacks directly on object-detection models, which results with a model-agnostic attack. We show that our proposed method works both digitally and physically. We compared our approach against four different black-box attacks with different configurations. Our approach outperformed all other approaches that were tested in our experiments by a large margin.



## **35. It Is All About Data: A Survey on the Effects of Data on Adversarial Robustness**

cs.LG

Accepted to ACM Computing Surveys, 40 pages, 24 figures

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2303.09767v3) [paper-pdf](http://arxiv.org/pdf/2303.09767v3)

**Authors**: Peiyu Xiong, Michael Tegegn, Jaskeerat Singh Sarin, Shubhraneel Pal, Julia Rubin

**Abstract**: Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to confuse the model into making a mistake. Such examples pose a serious threat to the applicability of machine-learning-based systems, especially in life- and safety-critical domains. To address this problem, the area of adversarial robustness investigates mechanisms behind adversarial attacks and defenses against these attacks. This survey reviews a particular subset of this literature that focuses on investigating properties of training data in the context of model robustness under evasion attacks. It first summarizes the main properties of data leading to adversarial vulnerability. It then discusses guidelines and techniques for improving adversarial robustness by enhancing the data representation and learning procedures, as well as techniques for estimating robustness guarantees given particular data. Finally, it discusses gaps of knowledge and promising future research directions in this area.



## **36. Defending Black-box Classifiers by Bayesian Boundary Correction**

cs.CV

arXiv admin note: text overlap with arXiv:2203.04713

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2306.16979v2) [paper-pdf](http://arxiv.org/pdf/2306.16979v2)

**Authors**: He Wang, Yunfeng Diao

**Abstract**: Classifiers based on deep neural networks have been recently challenged by Adversarial Attack, where the widely existing vulnerability has invoked the research in defending them from potential threats. Given a vulnerable classifier, existing defense methods are mostly white-box and often require re-training the victim under modified loss functions/training regimes. While the model/data/training specifics of the victim are usually unavailable to the user, re-training is unappealing, if not impossible for reasons such as limited computational resources. To this end, we propose a new black-box defense framework. It can turn any pre-trained classifier into a resilient one with little knowledge of the model specifics. This is achieved by new joint Bayesian treatments on the clean data, the adversarial examples and the classifier, for maximizing their joint probability. It is further equipped with a new post-train strategy which keeps the victim intact. We name our framework Bayesian Boundary Correction (BBC). BBC is a general and flexible framework that can easily adapt to different data types. We instantiate BBC for image classification and skeleton-based human activity recognition, for both static and dynamic data. Exhaustive evaluation shows that BBC has superior robustness and can enhance robustness without severely hurting the clean accuracy, compared with existing defense methods.



## **37. Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks**

cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10844v1) [paper-pdf](http://arxiv.org/pdf/2310.10844v1)

**Authors**: Erfan Shayegani, Md Abdullah Al Mamun, Yu Fu, Pedram Zaree, Yue Dong, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are swiftly advancing in architecture and capability, and as they integrate more deeply into complex systems, the urgency to scrutinize their security properties grows. This paper surveys research in the emerging interdisciplinary field of adversarial attacks on LLMs, a subfield of trustworthy ML, combining the perspectives of Natural Language Processing and Security. Prior work has shown that even safety-aligned LLMs (via instruction tuning and reinforcement learning through human feedback) can be susceptible to adversarial attacks, which exploit weaknesses and mislead AI systems, as evidenced by the prevalence of `jailbreak' attacks on models like ChatGPT and Bard. In this survey, we first provide an overview of large language models, describe their safety alignment, and categorize existing research based on various learning structures: textual-only attacks, multi-modal attacks, and additional attack methods specifically targeting complex systems, such as federated learning or multi-agent systems. We also offer comprehensive remarks on works that focus on the fundamental sources of vulnerabilities and potential defenses. To make this field more accessible to newcomers, we present a systematic review of existing works, a structured typology of adversarial attack concepts, and additional resources, including slides for presentations on related topics at the 62nd Annual Meeting of the Association for Computational Linguistics (ACL'24).



## **38. Regularization properties of adversarially-trained linear regression**

stat.ML

Accepted (spotlight) NeurIPS 2023; A preliminary version of this work  titled: "Surprises in adversarially-trained linear regression" was made  available under a different identifier: arXiv:2205.12695

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10807v1) [paper-pdf](http://arxiv.org/pdf/2310.10807v1)

**Authors**: Antônio H. Ribeiro, Dave Zachariah, Francis Bach, Thomas B. Schön

**Abstract**: State-of-the-art machine learning models can be vulnerable to very small input perturbations that are adversarially constructed. Adversarial training is an effective approach to defend against it. Formulated as a min-max problem, it searches for the best solution when the training data were corrupted by the worst-case attacks. Linear models are among the simple models where vulnerabilities can be observed and are the focus of our study. In this case, adversarial training leads to a convex optimization problem which can be formulated as the minimization of a finite sum. We provide a comparative analysis between the solution of adversarial training in linear regression and other regularization methods. Our main findings are that: (A) Adversarial training yields the minimum-norm interpolating solution in the overparameterized regime (more parameters than data), as long as the maximum disturbance radius is smaller than a threshold. And, conversely, the minimum-norm interpolator is the solution to adversarial training with a given radius. (B) Adversarial training can be equivalent to parameter shrinking methods (ridge regression and Lasso). This happens in the underparametrized region, for an appropriate choice of adversarial radius and zero-mean symmetrically distributed covariates. (C) For $\ell_\infty$-adversarial training -- as in square-root Lasso -- the choice of adversarial radius for optimal bounds does not depend on the additive noise variance. We confirm our theoretical findings with numerical examples.



## **39. Fast Adversarial Label-Flipping Attack on Tabular Data**

cs.LG

10 pages

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10744v1) [paper-pdf](http://arxiv.org/pdf/2310.10744v1)

**Authors**: Xinglong Chang, Gillian Dobbie, Jörg Wicker

**Abstract**: Machine learning models are increasingly used in fields that require high reliability such as cybersecurity. However, these models remain vulnerable to various attacks, among which the adversarial label-flipping attack poses significant threats. In label-flipping attacks, the adversary maliciously flips a portion of training labels to compromise the machine learning model. This paper raises significant concerns as these attacks can camouflage a highly skewed dataset as an easily solvable classification problem, often misleading machine learning practitioners into lower defenses and miscalculations of potential risks. This concern amplifies in tabular data settings, where identifying true labels requires expertise, allowing malicious label-flipping attacks to easily slip under the radar. To demonstrate this risk is inherited in the adversary's objective, we propose FALFA (Fast Adversarial Label-Flipping Attack), a novel efficient attack for crafting adversarial labels. FALFA is based on transforming the adversary's objective and employs linear programming to reduce computational complexity. Using ten real-world tabular datasets, we demonstrate FALFA's superior attack potential, highlighting the need for robust defenses against such threats.



## **40. Passive Inference Attacks on Split Learning via Adversarial Regularization**

cs.CR

16 pages, 16 figures

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10483v1) [paper-pdf](http://arxiv.org/pdf/2310.10483v1)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.



## **41. DANAA: Towards transferable attacks with double adversarial neuron attribution**

cs.CV

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10427v1) [paper-pdf](http://arxiv.org/pdf/2310.10427v1)

**Authors**: Zhibo Jin, Zhiyu Zhu, Xinyi Wang, Jiayu Zhang, Jun Shen, Huaming Chen

**Abstract**: While deep neural networks have excellent results in many fields, they are susceptible to interference from attacking samples resulting in erroneous judgments. Feature-level attacks are one of the effective attack types, which targets the learnt features in the hidden layers to improve its transferability across different models. Yet it is observed that the transferability has been largely impacted by the neuron importance estimation results. In this paper, a double adversarial neuron attribution attack method, termed `DANAA', is proposed to obtain more accurate feature importance estimation. In our method, the model outputs are attributed to the middle layer based on an adversarial non-linear path. The goal is to measure the weight of individual neurons and retain the features that are more important towards transferability. We have conducted extensive experiments on the benchmark datasets to demonstrate the state-of-the-art performance of our method. Our code is available at: https://github.com/Davidjinzb/DANAA



## **42. Privacy in Large Language Models: Attacks, Defenses and Future Directions**

cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10383v1) [paper-pdf](http://arxiv.org/pdf/2310.10383v1)

**Authors**: Haoran Li, Yulin Chen, Jinglong Luo, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, Yangqiu Song

**Abstract**: The advancement of large language models (LLMs) has significantly enhanced the ability to effectively tackle various downstream NLP tasks and unify these tasks into generative pipelines. On the one hand, powerful language models, trained on massive textual data, have brought unparalleled accessibility and usability for both models and users. On the other hand, unrestricted access to these models can also introduce potential malicious and unintentional privacy risks. Despite ongoing efforts to address the safety and privacy concerns associated with LLMs, the problem remains unresolved. In this paper, we provide a comprehensive analysis of the current privacy attacks targeting LLMs and categorize them according to the adversary's assumed capabilities to shed light on the potential vulnerabilities present in LLMs. Then, we present a detailed overview of prominent defense strategies that have been developed to counter these privacy attacks. Beyond existing works, we identify upcoming privacy concerns as LLMs evolve. Lastly, we point out several potential avenues for future exploration.



## **43. A White-Box False Positive Adversarial Attack Method on Contrastive Loss-Based Offline Handwritten Signature Verification Models**

cs.CV

8 pages, 2 figures

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2308.08925v2) [paper-pdf](http://arxiv.org/pdf/2308.08925v2)

**Authors**: Zhongliang Guo, Weiye Li, Yifei Qian, Ognjen Arandjelović, Lei Fang

**Abstract**: In this paper, we tackle the challenge of white-box false positive adversarial attacks on contrastive loss-based offline handwritten signature verification models. We propose a novel attack method that treats the attack as a style transfer between closely related but distinct writing styles. To guide the generation of deceptive images, we introduce two new loss functions that enhance the attack success rate by perturbing the Euclidean distance between the embedding vectors of the original and synthesized samples, while ensuring minimal perturbations by reducing the difference between the generated image and the original image. Our method demonstrates state-of-the-art performance in white-box attacks on contrastive loss-based offline handwritten signature verification models, as evidenced by our experiments. The key contributions of this paper include a novel false positive attack method, two new loss functions, effective style transfer in handwriting styles, and superior performance in white-box false positive attacks compared to other white-box attack methods.



## **44. A Non-monotonic Smooth Activation Function**

cs.LG

12 Pages

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10126v1) [paper-pdf](http://arxiv.org/pdf/2310.10126v1)

**Authors**: Koushik Biswas, Meghana Karri, Ulaş Bağcı

**Abstract**: Activation functions are crucial in deep learning models since they introduce non-linearity into the networks, allowing them to learn from errors and make adjustments, which is essential for learning complex patterns. The essential purpose of activation functions is to transform unprocessed input signals into significant output activations, promoting information transmission throughout the neural network. In this study, we propose a new activation function called Sqish, which is a non-monotonic and smooth function and an alternative to existing ones. We showed its superiority in classification, object detection, segmentation tasks, and adversarial robustness experiments. We got an 8.21% improvement over ReLU on the CIFAR100 dataset with the ShuffleNet V2 model in the FGSM adversarial attack. We also got a 5.87% improvement over ReLU on image classification on the CIFAR100 dataset with the ShuffleNet V2 model.



## **45. Evading Detection Actively: Toward Anti-Forensics against Forgery Localization**

cs.CV

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10036v1) [paper-pdf](http://arxiv.org/pdf/2310.10036v1)

**Authors**: Long Zhuo, Shenghai Luo, Shunquan Tan, Han Chen, Bin Li, Jiwu Huang

**Abstract**: Anti-forensics seeks to eliminate or conceal traces of tampering artifacts. Typically, anti-forensic methods are designed to deceive binary detectors and persuade them to misjudge the authenticity of an image. However, to the best of our knowledge, no attempts have been made to deceive forgery detectors at the pixel level and mis-locate forged regions. Traditional adversarial attack methods cannot be directly used against forgery localization due to the following defects: 1) they tend to just naively induce the target forensic models to flip their pixel-level pristine or forged decisions; 2) their anti-forensics performance tends to be severely degraded when faced with the unseen forensic models; 3) they lose validity once the target forensic models are retrained with the anti-forensics images generated by them. To tackle the three defects, we propose SEAR (Self-supErvised Anti-foRensics), a novel self-supervised and adversarial training algorithm that effectively trains deep-learning anti-forensic models against forgery localization. SEAR sets a pretext task to reconstruct perturbation for self-supervised learning. In adversarial training, SEAR employs a forgery localization model as a supervisor to explore tampering features and constructs a deep-learning concealer to erase corresponding traces. We have conducted largescale experiments across diverse datasets. The experimental results demonstrate that, through the combination of self-supervised learning and adversarial learning, SEAR successfully deceives the state-of-the-art forgery localization methods, as well as tackle the three defects regarding traditional adversarial attack methods mentioned above.



## **46. Black-box Targeted Adversarial Attack on Segment Anything (SAM)**

cs.CV

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10010v1) [paper-pdf](http://arxiv.org/pdf/2310.10010v1)

**Authors**: Sheng Zheng, Chaoning Zhang

**Abstract**: Deep recognition models are widely vulnerable to adversarial examples, which change the model output by adding quasi-imperceptible perturbation to the image input. Recently, Segment Anything Model (SAM) has emerged to become a popular foundation model in computer vision due to its impressive generalization to unseen data and tasks. Realizing flexible attacks on SAM is beneficial for understanding the robustness of SAM in the adversarial context. To this end, this work aims to achieve a targeted adversarial attack (TAA) on SAM. Specifically, under a certain prompt, the goal is to make the predicted mask of an adversarial example resemble that of a given target image. The task of TAA on SAM has been realized in a recent arXiv work in the white-box setup by assuming access to prompt and model, which is thus less practical. To address the issue of prompt dependence, we propose a simple yet effective approach by only attacking the image encoder. Moreover, we propose a novel regularization loss to enhance the cross-model transferability by increasing the feature dominance of adversarial images over random natural images. Extensive experiments verify the effectiveness of our proposed simple techniques to conduct a successful black-box TAA on SAM.



## **47. Towards Deep Learning Models Resistant to Transfer-based Adversarial Attacks via Data-centric Robust Learning**

cs.CR

9 pages

**SubmitDate**: 2023-10-15    [abs](http://arxiv.org/abs/2310.09891v1) [paper-pdf](http://arxiv.org/pdf/2310.09891v1)

**Authors**: Yulong Yang, Chenhao Lin, Xiang Ji, Qiwei Tian, Qian Li, Hongshan Yang, Zhibo Wang, Chao Shen

**Abstract**: Transfer-based adversarial attacks raise a severe threat to real-world deep learning systems since they do not require access to target models. Adversarial training (AT), which is recognized as the strongest defense against white-box attacks, has also guaranteed high robustness to (black-box) transfer-based attacks. However, AT suffers from heavy computational overhead since it optimizes the adversarial examples during the whole training process. In this paper, we demonstrate that such heavy optimization is not necessary for AT against transfer-based attacks. Instead, a one-shot adversarial augmentation prior to training is sufficient, and we name this new defense paradigm Data-centric Robust Learning (DRL). Our experimental results show that DRL outperforms widely-used AT techniques (e.g., PGD-AT, TRADES, EAT, and FAT) in terms of black-box robustness and even surpasses the top-1 defense on RobustBench when combined with diverse data augmentations and loss regularizations. We also identify other benefits of DRL, for instance, the model generalization capability and robust fairness.



## **48. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks**

cs.CV

**SubmitDate**: 2023-10-15    [abs](http://arxiv.org/abs/2306.09124v2) [paper-pdf](http://arxiv.org/pdf/2306.09124v2)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications, yet current research in this area is not satisfactory. In this paper, we propose DIFFender, a novel defense method that leverages a text-guided diffusion model to defend against adversarial patches. DIFFender includes two main stages: patch localization and patch restoration. In the localization stage, we find and exploit an intriguing property of the diffusion model to effectively identify the locations of adversarial patches. In the restoration stage, we employ the diffusion model to reconstruct the adversarial regions in the images while preserving the integrity of the visual content. Importantly, these two stages are carefully guided by a unified diffusion model, thus we can utilize the close interaction between them to improve the whole defense performance. Moreover, we propose a few-shot prompt-tuning algorithm to fine-tune the diffusion model, enabling the pre-trained diffusion model to easily adapt to the defense task. We conduct extensive experiments on the image classification and face recognition tasks, demonstrating that our proposed method exhibits superior robustness under strong adaptive attacks and generalizes well across various scenarios, diverse classifiers, and multiple patch attack methods.



## **49. Are Your Explanations Reliable? Investigating the Stability of LIME in Explaining Text Classifiers by Marrying XAI and Adversarial Attack**

cs.LG

14 pages, 6 figures. Replacement by the updated version to be  published in EMNLP 2023

**SubmitDate**: 2023-10-15    [abs](http://arxiv.org/abs/2305.12351v2) [paper-pdf](http://arxiv.org/pdf/2305.12351v2)

**Authors**: Christopher Burger, Lingwei Chen, Thai Le

**Abstract**: LIME has emerged as one of the most commonly referenced tools in explainable AI (XAI) frameworks that is integrated into critical machine learning applications--e.g., healthcare and finance. However, its stability remains little explored, especially in the context of text data, due to the unique text-space constraints. To address these challenges, in this paper, we first evaluate the inherent instability of LIME on text data to establish a baseline, and then propose a novel algorithm XAIFooler to perturb text inputs and manipulate explanations that casts investigation on the stability of LIME as a text perturbation optimization problem. XAIFooler conforms to the constraints to preserve text semantics and original prediction with small perturbations, and introduces Rank-biased Overlap (RBO) as a key part to guide the optimization of XAIFooler that satisfies all the requirements for explanation similarity measure. Extensive experiments on real-world text datasets demonstrate that XAIFooler significantly outperforms all baselines by large margins in its ability to manipulate LIME's explanations with high semantic preservability.



## **50. Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game**

cs.GT

**SubmitDate**: 2023-10-15    [abs](http://arxiv.org/abs/2305.12872v2) [paper-pdf](http://arxiv.org/pdf/2305.12872v2)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Ruixiao Xu, Xin Yu, Jiakai Wang, Aishan Liu, Yaodong Yang, Xianglong Liu

**Abstract**: In this study, we explore the robustness of cooperative multi-agent reinforcement learning (c-MARL) against Byzantine failures, where any agent can enact arbitrary, worst-case actions due to malfunction or adversarial attack. To address the uncertainty that any agent can be adversarial, we propose a Bayesian Adversarial Robust Dec-POMDP (BARDec-POMDP) framework, which views Byzantine adversaries as nature-dictated types, represented by a separate transition. This allows agents to learn policies grounded on their posterior beliefs about the type of other agents, fostering collaboration with identified allies and minimizing vulnerability to adversarial manipulation. We define the optimal solution to the BARDec-POMDP as an ex post robust Bayesian Markov perfect equilibrium, which we proof to exist and weakly dominates the equilibrium of previous robust MARL approaches. To realize this equilibrium, we put forward a two-timescale actor-critic algorithm with almost sure convergence under specific conditions. Experimentation on matrix games, level-based foraging and StarCraft II indicate that, even under worst-case perturbations, our method successfully acquires intricate micromanagement skills and adaptively aligns with allies, demonstrating resilience against non-oblivious adversaries, random allies, observation-based attacks, and transfer-based attacks.



