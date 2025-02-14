# Latest Adversarial Attack Papers
**update at 2025-02-14 10:56:58**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. On the Importance of Backbone to the Adversarial Robustness of Object Detectors**

cs.CV

Accepted by IEEE TIFS

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2305.17438v2) [paper-pdf](http://arxiv.org/pdf/2305.17438v2)

**Authors**: Xiao Li, Hang Chen, Xiaolin Hu

**Abstract**: Object detection is a critical component of various security-sensitive applications, such as autonomous driving and video surveillance. However, existing object detectors are vulnerable to adversarial attacks, which poses a significant challenge to their reliability and security. Through experiments, first, we found that existing works on improving the adversarial robustness of object detectors give a false sense of security. Second, we found that adversarially pre-trained backbone networks were essential for enhancing the adversarial robustness of object detectors. We then proposed a simple yet effective recipe for fast adversarial fine-tuning on object detectors with adversarially pre-trained backbones. Without any modifications to the structure of object detectors, our recipe achieved significantly better adversarial robustness than previous works. Finally, we explored the potential of different modern object detector designs for improving adversarial robustness with our recipe and demonstrated interesting findings, which inspired us to design state-of-the-art (SOTA) robust detectors. Our empirical results set a new milestone for adversarially robust object detection. Code and trained checkpoints are available at https://github.com/thu-ml/oddefense.



## **2. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

cs.LG

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2408.00315v3) [paper-pdf](http://arxiv.org/pdf/2408.00315v3)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.



## **3. Wasserstein distributional adversarial training for deep neural networks**

cs.LG

15 pages, 4 figures

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09352v1) [paper-pdf](http://arxiv.org/pdf/2502.09352v1)

**Authors**: Xingjian Bai, Guangyi He, Yifan Jiang, Jan Obloj

**Abstract**: Design of adversarial attacks for deep neural networks, as well as methods of adversarial training against them, are subject of intense research. In this paper, we propose methods to train against distributional attack threats, extending the TRADES method used for pointwise attacks. Our approach leverages recent contributions and relies on sensitivity analysis for Wasserstein distributionally robust optimization problems. We introduce an efficient fine-tuning method which can be deployed on a previously trained model. We test our methods on a range of pre-trained models on RobustBench. These experimental results demonstrate the additional training enhances Wasserstein distributional robustness, while maintaining original levels of pointwise robustness, even for already very successful networks. The improvements are less marked for models pre-trained using huge synthetic datasets of 20-100M images. However, remarkably, sometimes our methods are still able to improve their performance even when trained using only the original training dataset (50k images).



## **4. LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection**

cs.LG

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09271v1) [paper-pdf](http://arxiv.org/pdf/2502.09271v1)

**Authors**: Wenlun Zhang, Enyan Dai, Kentaro Yoshioka

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.



## **5. FLAME: Flexible LLM-Assisted Moderation Engine**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09175v1) [paper-pdf](http://arxiv.org/pdf/2502.09175v1)

**Authors**: Ivan Bakulin, Ilia Kopanichuk, Iaroslav Bespalov, Nikita Radchenko, Vladimir Shaposhnikov, Dmitry Dylov, Ivan Oseledets

**Abstract**: The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs.



## **6. Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks**

cs.CV

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09110v1) [paper-pdf](http://arxiv.org/pdf/2502.09110v1)

**Authors**: Eylon Mizrahi, Raz Lapid, Moshe Sipper

**Abstract**: Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.



## **7. Universal Adversarial Attack on Aligned Multimodal LLMs**

cs.AI

Added an affiliation

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.07987v2) [paper-pdf](http://arxiv.org/pdf/2502.07987v2)

**Authors**: Temurbek Rahmatullaev, Polina Druzhinina, Matvey Mikhalchuk, Andrey Kuznetsov, Anton Razzhigaev

**Abstract**: We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.



## **8. RLSA-PFL: Robust Lightweight Secure Aggregation with Model Inconsistency Detection in Privacy-Preserving Federated Learning**

cs.CR

16 pages, 10 Figures

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.08989v1) [paper-pdf](http://arxiv.org/pdf/2502.08989v1)

**Authors**: Nazatul H. Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba

**Abstract**: Federated Learning (FL) allows users to collaboratively train a global machine learning model by sharing local model only, without exposing their private data to a central server. This distributed learning is particularly appealing in scenarios where data privacy is crucial, and it has garnered substantial attention from both industry and academia. However, studies have revealed privacy vulnerabilities in FL, where adversaries can potentially infer sensitive information from the shared model parameters. In this paper, we present an efficient masking-based secure aggregation scheme utilizing lightweight cryptographic primitives to mitigate privacy risks. Our scheme offers several advantages over existing methods. First, it requires only a single setup phase for the entire FL training session, significantly reducing communication overhead. Second, it minimizes user-side overhead by eliminating the need for user-to-user interactions, utilizing an intermediate server layer and a lightweight key negotiation method. Third, the scheme is highly resilient to user dropouts, and the users can join at any FL round. Fourth, it can detect and defend against malicious server activities, including recently discovered model inconsistency attacks. Finally, our scheme ensures security in both semi-honest and malicious settings. We provide security analysis to formally prove the robustness of our approach. Furthermore, we implemented an end-to-end prototype of our scheme. We conducted comprehensive experiments and comparisons, which show that it outperforms existing solutions in terms of communication and computation overhead, functionality, and security.



## **9. An Engorgio Prompt Makes Large Language Model Babble on**

cs.CR

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2412.19394v2) [paper-pdf](http://arxiv.org/pdf/2412.19394v2)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu, Han Qiu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is released at: https://github.com/jianshuod/Engorgio-prompt.



## **10. Siren Song: Manipulating Pose Estimation in XR Headsets Using Acoustic Attacks**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.08865v1) [paper-pdf](http://arxiv.org/pdf/2502.08865v1)

**Authors**: Zijian Huang, Yicheng Zhang, Sophie Chen, Nael Abu-Ghazaleh, Jiasi Chen

**Abstract**: Extended Reality (XR) experiences involve interactions between users, the real world, and virtual content. A key step to enable these experiences is the XR headset sensing and estimating the user's pose in order to accurately place and render virtual content in the real world. XR headsets use multiple sensors (e.g., cameras, inertial measurement unit) to perform pose estimation and improve its robustness, but this provides an attack surface for adversaries to interfere with the pose estimation process. In this paper, we create and study the effects of acoustic attacks that create false signals in the inertial measurement unit (IMU) on XR headsets, leading to adverse downstream effects on XR applications. We generate resonant acoustic signals on a HoloLens 2 and measure the resulting perturbations in the IMU readings, and also demonstrate both fine-grained and coarse attacks on the popular ORB-SLAM3 and an open-source XR system (ILLIXR). With the knowledge gleaned from attacking these open-source frameworks, we demonstrate four end-to-end proof-of-concept attacks on a HoloLens 2: manipulating user input, clickjacking, zone invasion, and denial of user interaction. Our experiments show that current commercial XR headsets are susceptible to acoustic attacks, raising concerns for their security.



## **11. ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech**

eess.AS

Database link: https://zenodo.org/records/14498691, Database mirror  link: https://huggingface.co/datasets/jungjee/asvspoof5, ASVspoof 5 Challenge  Workshop Proceeding: https://www.isca-archive.org/asvspoof_2024/index.html

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.08857v1) [paper-pdf](http://arxiv.org/pdf/2502.08857v1)

**Authors**: Xin Wang, Héctor Delgado, Hemlata Tak, Jee-weon Jung, Hye-jin Shim, Massimiliano Todisco, Ivan Kukanov, Xuechen Liu, Md Sahidullah, Tomi Kinnunen, Nicholas Evans, Kong Aik Lee, Junichi Yamagishi, Myeonghun Jeong, Ge Zhu, Yongyi Zang, You Zhang, Soumi Maiti, Florian Lux, Nicolas Müller, Wangyou Zhang, Chengzhe Sun, Shuwei Hou, Siwei Lyu, Sébastien Le Maguer, Cheng Gong, Hanjie Guo, Liping Chen, Vishwanath Singh

**Abstract**: ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake attacks as well as the design of detection solutions. We introduce the ASVspoof 5 database which is generated in crowdsourced fashion from data collected in diverse acoustic conditions (cf. studio-quality data for earlier ASVspoof databases) and from ~2,000 speakers (cf. ~100 earlier). The database contains attacks generated with 32 different algorithms, also crowdsourced, and optimised to varying degrees using new surrogate detection models. Among them are attacks generated with a mix of legacy and contemporary text-to-speech synthesis and voice conversion models, in addition to adversarial attacks which are incorporated for the first time. ASVspoof 5 protocols comprise seven speaker-disjoint partitions. They include two distinct partitions for the training of different sets of attack models, two more for the development and evaluation of surrogate detection models, and then three additional partitions which comprise the ASVspoof 5 training, development and evaluation sets. An auxiliary set of data collected from an additional 30k speakers can also be used to train speaker encoders for the implementation of attack algorithms. Also described herein is an experimental validation of the new ASVspoof 5 database using a set of automatic speaker verification and spoof/deepfake baseline detectors. With the exception of protocols and tools for the generation of spoofed/deepfake speech, the resources described in this paper, already used by participants of the ASVspoof 5 challenge in 2024, are now all freely available to the community.



## **12. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2410.02890v3) [paper-pdf](http://arxiv.org/pdf/2410.02890v3)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.



## **13. Bankrupting DoS Attackers**

cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2205.08287v4) [paper-pdf](http://arxiv.org/pdf/2205.08287v4)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstract**: Can we make a denial-of-service attacker pay more than the server and honest clients? Consider a model where a server sees a stream of jobs sent by either honest clients or an adversary. The server sets a price for servicing each job with the aid of an estimator, which provides approximate statistical information about the distribution of previously occurring good jobs.   We describe and analyze pricing algorithms for the server under different models of synchrony, with total cost parameterized by the accuracy of the estimator. Given a reasonably accurate estimator, the algorithm's cost provably grows more slowly than the attacker's cost, as the attacker's cost grows large. Additionally, we prove a lower bound, showing that our pricing algorithm yields asymptotically tight results when the estimator is accurate within constant factors.



## **14. Extreme vulnerability to intruder attacks destabilizes network dynamics**

nlin.AO

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08552v1) [paper-pdf](http://arxiv.org/pdf/2502.08552v1)

**Authors**: Amirhossein Nazerian, Sahand Tangerami, Malbor Asllani, David Phillips, Hernan Makse, Francesco Sorrentino

**Abstract**: Consensus, synchronization, formation control, and power grid balance are all examples of virtuous dynamical states that may arise in networks. Here we focus on how such states can be destabilized from a fundamental perspective; namely, we address the question of how one or a few intruder agents within an otherwise functioning network may compromise its dynamics. We show that a single adversarial node coupled via adversarial couplings to one or more other nodes is sufficient to destabilize the entire network, which we prove to be more efficient than targeting multiple nodes. Then, we show that concentrating the attack on a single low-indegree node induces the greatest instability, challenging the common assumption that hubs are the most critical nodes. This leads to a new characterization of the vulnerability of a node, which contrasts with previous work, and identifies low-indegree nodes (as opposed to the hubs) as the most vulnerable components of a network. Our results are derived for linear systems but hold true for nonlinear networks, including those described by the Kuramoto model. Finally, we derive scaling laws showing that larger networks are less susceptible, on average, to single-node attacks. Overall, these findings highlight an intrinsic vulnerability of technological systems such as autonomous networks, sensor networks, power grids, and the internet of things, which also extend to the realm of complex social and biological networks.



## **15. The Impact of Logic Locking on Confidentiality: An Automated Evaluation**

cs.CR

8 pages, accepted at 26th International Symposium on Quality  Electronic Design (ISQED'25)

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.01240v2) [paper-pdf](http://arxiv.org/pdf/2502.01240v2)

**Authors**: Lennart M. Reimann, Evgenii Rezunov, Dominik Germek, Luca Collini, Christian Pilato, Ramesh Karri, Rainer Leupers

**Abstract**: Logic locking secures hardware designs in untrusted foundries by incorporating key-driven gates to obscure the original blueprint. While this method safeguards the integrated circuit from malicious alterations during fabrication, its influence on data confidentiality during runtime has been ignored. In this study, we employ path sensitization to formally examine the impact of logic locking on confidentiality. By applying three representative logic locking mechanisms on open-source cryptographic benchmarks, we utilize an automatic test pattern generation framework to evaluate the effect of locking on cryptographic encryption keys and sensitive data signals. Our analysis reveals that logic locking can inadvertently cause sensitive data leakage when incorrect logic locking keys are used. We show that a single malicious logic locking key can expose over 70% of an encryption key. If an adversary gains control over other inputs, the entire encryption key can be compromised. This research uncovers a significant security vulnerability in logic locking and emphasizes the need for comprehensive security assessments that extend beyond key-recovery attacks.



## **16. AdvSwap: Covert Adversarial Perturbation with High Frequency Info-swapping for Autonomous Driving Perception**

cs.CV

27th IEEE International Conference on Intelligent Transportation  Systems (ITSC)

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08374v1) [paper-pdf](http://arxiv.org/pdf/2502.08374v1)

**Authors**: Yuanhao Huang, Qinfan Zhang, Jiandong Xing, Mengyue Cheng, Haiyang Yu, Yilong Ren, Xiao Xiong

**Abstract**: Perception module of Autonomous vehicles (AVs) are increasingly susceptible to be attacked, which exploit vulnerabilities in neural networks through adversarial inputs, thereby compromising the AI safety. Some researches focus on creating covert adversarial samples, but existing global noise techniques are detectable and difficult to deceive the human visual system. This paper introduces a novel adversarial attack method, AdvSwap, which creatively utilizes wavelet-based high-frequency information swapping to generate covert adversarial samples and fool the camera. AdvSwap employs invertible neural network for selective high-frequency information swapping, preserving both forward propagation and data integrity. The scheme effectively removes the original label data and incorporates the guidance image data, producing concealed and robust adversarial samples. Experimental evaluations and comparisons on the GTSRB and nuScenes datasets demonstrate that AdvSwap can make concealed attacks on common traffic targets. The generates adversarial samples are also difficult to perceive by humans and algorithms. Meanwhile, the method has strong attacking robustness and attacking transferability.



## **17. TASAR: Transfer-based Attack on Skeletal Action Recognition**

cs.CV

Accepted in ICLR 2025

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2409.02483v5) [paper-pdf](http://arxiv.org/pdf/2409.02483v5)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Ajian Liu, Xiaoshuai Hao, Xingxing Wei, Meng Wang, He Wang

**Abstract**: Skeletal sequence data, as a widely employed representation of human actions, are crucial in Human Activity Recognition (HAR). Recently, adversarial attacks have been proposed in this area, which exposes potential security concerns, and more importantly provides a good tool for model robustness test. Within this research, transfer-based attack is an important tool as it mimics the real-world scenario where an attacker has no knowledge of the target model, but is under-explored in Skeleton-based HAR (S-HAR). Consequently, existing S-HAR attacks exhibit weak adversarial transferability and the reason remains largely unknown. In this paper, we investigate this phenomenon via the characterization of the loss function. We find that one prominent indicator of poor transferability is the low smoothness of the loss function. Led by this observation, we improve the transferability by properly smoothening the loss when computing the adversarial examples. This leads to the first Transfer-based Attack on Skeletal Action Recognition, TASAR. TASAR explores the smoothened model posterior of pre-trained surrogates, which is achieved by a new post-train Dual Bayesian optimization strategy. Furthermore, unlike existing transfer-based methods which overlook the temporal coherence within sequences, TASAR incorporates motion dynamics into the Bayesian attack, effectively disrupting the spatial-temporal coherence of S-HARs. For exhaustive evaluation, we build the first large-scale robust S-HAR benchmark, comprising 7 S-HAR models, 10 attack methods, 3 S-HAR datasets and 2 defense models. Extensive results demonstrate the superiority of TASAR. Our benchmark enables easy comparisons for future studies, with the code available in the https://github.com/yunfengdiao/Skeleton-Robustness-Benchmark.



## **18. RIDA: A Robust Attack Framework on Incomplete Graphs**

cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2407.18170v3) [paper-pdf](http://arxiv.org/pdf/2407.18170v3)

**Authors**: Jianke Yu, Hanchen Wang, Chen Chen, Xiaoyang Wang, Lu Qin, Wenjie Zhang, Ying Zhang, Xijuan Liu

**Abstract**: Graph Neural Networks (GNNs) are vital in data science but are increasingly susceptible to adversarial attacks. To help researchers develop more robust GNN models, it's essential to focus on designing strong attack models as foundational benchmarks and guiding references. Among adversarial attacks, gray-box poisoning attacks are noteworthy due to their effectiveness and fewer constraints. These attacks exploit GNNs' need for retraining on updated data, thereby impacting their performance by perturbing these datasets. However, current research overlooks the real-world scenario of incomplete graphs. To address this gap, we introduce the Robust Incomplete Deep Attack Framework (RIDA). It is the first algorithm for robust gray-box poisoning attacks on incomplete graphs. The approach innovatively aggregates distant vertex information and ensures powerful data utilization. Extensive tests against 9 SOTA baselines on 3 real-world datasets demonstrate that RIDA's superiority in handling incompleteness and high attack performance on the incomplete graph.



## **19. Time-based GNSS attack detection**

cs.CR

IEEE Transactions on Aerospace and Electronic Systems (Early Access)

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.03868v2) [paper-pdf](http://arxiv.org/pdf/2502.03868v2)

**Authors**: Marco Spanghero, Panos Papadimitratos

**Abstract**: To safeguard Civilian Global Navigation Satellite Systems (GNSS) external information available to the platform encompassing the GNSS receiver can be used to detect attacks. Cross-checking the GNSS-provided time against alternative multiple trusted time sources can lead to attack detection aiming at controlling the GNSS receiver time. Leveraging external, network-connected secure time providers and onboard clock references, we achieve detection even under fine-grained time attacks. We provide an extensive evaluation of our multi-layered defense against adversaries mounting attacks against the GNSS receiver along with controlling the network link. We implement adversaries spanning from simplistic spoofers to advanced ones synchronized with the GNSS constellation. We demonstrate attack detection is possible in all tested cases (sharp discontinuity, smooth take-over, and coordinated network manipulation) without changes to the structure of the GNSS receiver. Leveraging the diversity of the reference time sources, detection of take-over time push as low as 150us is possible. Smooth take-overs forcing variations as low as 30ns are also detected based on on-board precision oscillators. The method (and thus the evaluation) is largely agnostic to the satellite constellation and the attacker type, making time-based data validation of GNSS information compatible with existing receivers and readily deployable.



## **20. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2409.20002v3) [paper-pdf](http://arxiv.org/pdf/2409.20002v3)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.



## **21. Safety at Scale: A Comprehensive Survey of Large Model Safety**

cs.CR

47 pages, 3 figures, 11 tables GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.05206v2) [paper-pdf](http://arxiv.org/pdf/2502.05206v2)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.



## **22. Optimizing Robustness and Accuracy in Mixture of Experts: A Dual-Model Approach**

cs.LG

10 pages, 3 figures, submitted to ICML 2025 (under review)

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.06832v2) [paper-pdf](http://arxiv.org/pdf/2502.06832v2)

**Authors**: Xu Zhang, Kaidi Xu, Ziqing Hu, Ren Wang

**Abstract**: Mixture of Experts (MoE) have shown remarkable success in leveraging specialized expert networks for complex machine learning tasks. However, their susceptibility to adversarial attacks presents a critical challenge for deployment in robust applications. This paper addresses the critical question of how to incorporate robustness into MoEs while maintaining high natural accuracy. We begin by analyzing the vulnerability of MoE components, finding that expert networks are notably more susceptible to adversarial attacks than the router. Based on this insight, we propose a targeted robust training technique that integrates a novel loss function to enhance the adversarial robustness of MoE, requiring only the robustification of one additional expert without compromising training or inference efficiency. Building on this, we introduce a dual-model strategy that linearly combines a standard MoE model with our robustified MoE model using a smoothing parameter. This approach allows for flexible control over the robustness-accuracy trade-off. We further provide theoretical foundations by deriving certified robustness bounds for both the single MoE and the dual-model. To push the boundaries of robustness and accuracy, we propose a novel joint training strategy JTDMoE for the dual-model. This joint training enhances both robustness and accuracy beyond what is achievable with separate models. Experimental results on CIFAR-10 and TinyImageNet datasets using ResNet18 and Vision Transformer (ViT) architectures demonstrate the effectiveness of our proposed methods.



## **23. Real-Time Privacy Risk Measurement with Privacy Tokens for Gradient Leakage**

cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.02913v4) [paper-pdf](http://arxiv.org/pdf/2502.02913v4)

**Authors**: Jiayang Meng, Tao Huang, Hong Chen, Xin Shi, Qingyu Huang, Chen Hou

**Abstract**: The widespread deployment of deep learning models in privacy-sensitive domains has amplified concerns regarding privacy risks, particularly those stemming from gradient leakage during training. Current privacy assessments primarily rely on post-training attack simulations. However, these methods are inherently reactive, unable to encompass all potential attack scenarios, and often based on idealized adversarial assumptions. These limitations underscore the need for proactive approaches to privacy risk assessment during the training process. To address this gap, we propose the concept of privacy tokens, which are derived directly from private gradients during training. Privacy tokens encapsulate gradient features and, when combined with data features, offer valuable insights into the extent of private information leakage from training data, enabling real-time measurement of privacy risks without relying on adversarial attack simulations. Additionally, we employ Mutual Information (MI) as a robust metric to quantify the relationship between training data and gradients, providing precise and continuous assessments of privacy leakage throughout the training process. Extensive experiments validate our framework, demonstrating the effectiveness of privacy tokens and MI in identifying and quantifying privacy risks. This proactive approach marks a significant advancement in privacy monitoring, promoting the safer deployment of deep learning models in sensitive applications.



## **24. Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**

cs.CR

14 pages, 4 figures, conference

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2501.02629v2) [paper-pdf](http://arxiv.org/pdf/2501.02629v2)

**Authors**: Yang Ouyang, Hengrui Gu, Shuhang Lin, Wenyue Hua, Jie Peng, Bhavya Kailkhura, Meijun Gao, Tianlong Chen, Kaixiong Zhou

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse applications, including chatbot assistants and code generation, aligning their behavior with safety and ethical standards has become paramount. However, jailbreak attacks, which exploit vulnerabilities to elicit unintended or harmful outputs, threaten LLMs' safety significantly. In this paper, we introduce Layer-AdvPatcher, a novel methodology designed to defend against jailbreak attacks by utilizing an unlearning strategy to patch specific layers within LLMs through self-augmented datasets. Our insight is that certain layer(s), tend to produce affirmative tokens when faced with harmful prompts. By identifying these layers and adversarially exposing them to generate more harmful data, one can understand their inherent and diverse vulnerabilities to attacks. With these exposures, we then "unlearn" these issues, reducing the impact of affirmative tokens and hence minimizing jailbreak risks while keeping the model's responses to safe queries intact. We conduct extensive experiments on two models, four benchmark datasets, and multiple state-of-the-art jailbreak attacks to demonstrate the efficacy of our approach. Results indicate that our framework reduces the harmfulness and attack success rate of jailbreak attacks without compromising utility for benign queries compared to recent defense methods. Our code is publicly available at: https://github.com/oyy2000/LayerAdvPatcher



## **25. Pseudorandom Permutations from Random Reversible Circuits**

cs.CC

Merged with arXiv:2409.14614; for merged paper see arXiv:2502.07159.  A previous version of one of the merged components of this paper contained  candidate constructions of computationally pseudorandom permutations from  one-way functions. There was an error in the proof of security, and we have  withdrawn this result

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2404.14648v4) [paper-pdf](http://arxiv.org/pdf/2404.14648v4)

**Authors**: William He, Ryan O'Donnell

**Abstract**: We study pseudorandomness properties of permutations on $\{0,1\}^n$ computed by random circuits made from reversible $3$-bit gates (permutations on $\{0,1\}^3$). Our main result is that a random circuit of depth $n \cdot \tilde{O}(k^2)$, with each layer consisting of $\approx n/3$ random gates in a fixed nearest-neighbor architecture, yields almost $k$-wise independent permutations. The main technical component is showing that the Markov chain on $k$-tuples of $n$-bit strings induced by a single random $3$-bit nearest-neighbor gate has spectral gap at least $1/n \cdot \tilde{O}(k)$. This improves on the original work of Gowers [Gowers96], who showed a gap of $1/\mathrm{poly}(n,k)$ for one random gate (with non-neighboring inputs); and, on subsequent work [HMMR05,BH08] improving the gap to $\Omega(1/n^2k)$ in the same setting.   From the perspective of cryptography, our result can be seen as a particularly simple/practical block cipher construction that gives provable statistical security against attackers with access to $k$~input-output pairs within few rounds. We also show that the Luby--Rackoff construction of pseudorandom permutations from pseudorandom functions can be implemented with reversible circuits. From this, we make progress on the complexity of the Minimum Reversible Circuit Size Problem (MRCSP), showing that block ciphers of fixed polynomial size are computationally secure against arbitrary polynomial-time adversaries, assuming the existence of one-way functions (OWFs).



## **26. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

cs.CV

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08079v1) [paper-pdf](http://arxiv.org/pdf/2502.08079v1)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.



## **27. Cascading Bandits Robust to Adversarial Corruptions**

cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08077v1) [paper-pdf](http://arxiv.org/pdf/2502.08077v1)

**Authors**: Jize Xie, Cheng Chen, Zhiyong Wang, Shuai Li

**Abstract**: Online learning to rank sequentially recommends a small list of items to users from a large candidate set and receives the users' click feedback. In many real-world scenarios, users browse the recommended list in order and click the first attractive item without checking the rest. Such behaviors are usually formulated as the cascade model. Many recent works study algorithms for cascading bandits, an online learning to rank framework in the cascade model. However, the performance of existing methods may drop significantly if part of the user feedback is adversarially corrupted (e.g., click fraud). In this work, we study how to resist adversarial corruptions in cascading bandits. We first formulate the ``\textit{Cascading Bandits with Adversarial Corruptions}" (CBAC) problem, which assumes that there is an adaptive adversary that may manipulate the user feedback. Then we propose two robust algorithms for this problem, which assume the corruption level is known and agnostic, respectively. We show that both algorithms can achieve logarithmic regret when the algorithm is not under attack, and the regret increases linearly with the corruption level. The experimental results also verify the robustness of our methods.



## **28. RESIST: Resilient Decentralized Learning Using Consensus Gradient Descent**

cs.LG

preprint of a journal paper; 100 pages and 17 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07977v1) [paper-pdf](http://arxiv.org/pdf/2502.07977v1)

**Authors**: Cheng Fang, Rishabh Dixit, Waheed U. Bajwa, Mert Gurbuzbalaban

**Abstract**: Empirical risk minimization (ERM) is a cornerstone of modern machine learning (ML), supported by advances in optimization theory that ensure efficient solutions with provable algorithmic convergence rates, which measure the speed at which optimization algorithms approach a solution, and statistical learning rates, which characterize how well the solution generalizes to unseen data. Privacy, memory, computational, and communications constraints increasingly necessitate data collection, processing, and storage across network-connected devices. In many applications, these networks operate in decentralized settings where a central server cannot be assumed, requiring decentralized ML algorithms that are both efficient and resilient. Decentralized learning, however, faces significant challenges, including an increased attack surface for adversarial interference during decentralized learning processes. This paper focuses on the man-in-the-middle (MITM) attack, which can cause models to deviate significantly from their intended ERM solutions. To address this challenge, we propose RESIST (Resilient dEcentralized learning using conSensus gradIent deScenT), an optimization algorithm designed to be robust against adversarially compromised communication links. RESIST achieves algorithmic and statistical convergence for strongly convex, Polyak-Lojasiewicz, and nonconvex ERM problems. Experimental results demonstrate the robustness and scalability of RESIST for real-world decentralized learning in adversarial environments.



## **29. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

cs.CR

Accepted by ICLR 2025

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2409.11295v4) [paper-pdf](http://arxiv.org/pdf/2409.11295v4)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have demonstrated remarkable potential in autonomously completing a wide range of tasks on real websites, significantly boosting human productivity. However, web tasks, such as booking flights, usually involve users' PII, which may be exposed to potential privacy risks if web agents accidentally interact with compromised websites, a scenario that remains largely unexplored in the literature. In this work, we narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a realistic threat model for attacks on the website, where we consider two adversarial targets: stealing users' specific PII or the entire user request. Then, we propose a novel attack method, termed Environmental Injection Attack (EIA). EIA injects malicious content designed to adapt well to environments where the agents operate and our work instantiates EIA specifically for privacy scenarios in web environments. We collect 177 action steps that involve diverse PII categories on realistic websites from the Mind2Web, and conduct experiments using one of the most capable generalist web agent frameworks to date. The results demonstrate that EIA achieves up to 70% ASR in stealing specific PII and 16% ASR for full user request. Additionally, by accessing the stealthiness and experimenting with a defensive system prompt, we indicate that EIA is hard to detect and mitigate. Notably, attacks that are not well adapted for a webpage can be detected via human inspection, leading to our discussion about the trade-off between security and autonomy. However, extra attackers' efforts can make EIA seamlessly adapted, rendering such supervision ineffective. Thus, we further discuss the defenses at the pre- and post-deployment stages of the websites without relying on human supervision and call for more advanced defense strategies.



## **30. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

cs.AI

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2407.00075v3) [paper-pdf](http://arxiv.org/pdf/2407.00075v3)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.



## **31. Approximate Energetic Resilience of Nonlinear Systems under Partial Loss of Control Authority**

math.OC

20 pages, 1 figure

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07603v1) [paper-pdf](http://arxiv.org/pdf/2502.07603v1)

**Authors**: Ram Padmanabhan, Melkior Ornik

**Abstract**: In this paper, we quantify the resilience of nonlinear dynamical systems by studying the increased energy used by all inputs of a system that suffers a partial loss of control authority, either through actuator malfunctions or through adversarial attacks. To quantify the maximal increase in energy, we introduce the notion of an energetic resilience metric. Prior work in this particular setting considers only simple linear models and not general nonlinear dynamical systems. We first characterize the mean value of the control signal in both the nominal and malfunctioning systems, which allows us to approximate the energy in the control. We then obtain a worst-case approximation of this energy for the malfunctioning system, over all malfunctioning inputs. Based on this approximation, we derive bounds on the energetic resilience metric when control authority is lost over one actuator. A simulation example on an academic nonlinear system demonstrates that the metric is useful in quantifying the resilience of the system without significant conservatism, despite the approximations used in obtaining control energies.



## **32. Efficient Image-to-Image Diffusion Classifier for Adversarial Robustness**

cs.CV

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2408.08502v2) [paper-pdf](http://arxiv.org/pdf/2408.08502v2)

**Authors**: Hefei Mei, Minjing Dong, Chang Xu

**Abstract**: Diffusion models (DMs) have demonstrated great potential in the field of adversarial robustness, where DM-based defense methods can achieve superior defense capability without adversarial training. However, they all require huge computational costs due to the usage of large-scale pre-trained DMs, making it difficult to conduct full evaluation under strong attacks and compare with traditional CNN-based methods. Simply reducing the network size and timesteps in DMs could significantly harm the image generation quality, which invalidates previous frameworks. To alleviate this issue, we redesign the diffusion framework from generating high-quality images to predicting distinguishable image labels. Specifically, we employ an image translation framework to learn many-to-one mapping from input samples to designed orthogonal image labels. Based on this framework, we introduce an efficient Image-to-Image diffusion classifier with a pruned U-Net structure and reduced diffusion timesteps. Besides the framework, we redesign the optimization objective of DMs to fit the target of image classification, where a new classification loss is incorporated in the DM-based image translation framework to distinguish the generated label from those of other classes. We conduct sufficient evaluations of the proposed classifier under various attacks on popular benchmarks. Extensive experiments show that our method achieves better adversarial robustness with fewer computational costs than DM-based and CNN-based methods. The code is available at https://github.com/hfmei/IDC



## **33. RoMA: Robust Malware Attribution via Byte-level Adversarial Training with Global Perturbations and Adversarial Consistency Regularization**

cs.CR

13 pages, 4 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07492v1) [paper-pdf](http://arxiv.org/pdf/2502.07492v1)

**Authors**: Yuxia Sun, Huihong Chen, Jingcai Guo, Aoxiang Sun, Zhetao Li, Haolin Liu

**Abstract**: Attributing APT (Advanced Persistent Threat) malware to their respective groups is crucial for threat intelligence and cybersecurity. However, APT adversaries often conceal their identities, rendering attribution inherently adversarial. Existing machine learning-based attribution models, while effective, remain highly vulnerable to adversarial attacks. For example, the state-of-the-art byte-level model MalConv sees its accuracy drop from over 90% to below 2% under PGD (projected gradient descent) attacks. Existing gradient-based adversarial training techniques for malware detection or image processing were applied to malware attribution in this study, revealing that both robustness and training efficiency require significant improvement. To address this, we propose RoMA, a novel single-step adversarial training approach that integrates global perturbations to generate enhanced adversarial samples and employs adversarial consistency regularization to improve representation quality and resilience. A novel APT malware dataset named AMG18, with diverse samples and realistic class imbalances, is introduced for evaluation. Extensive experiments show that RoMA significantly outperforms seven competing methods in both adversarial robustness (e.g., achieving over 80% robust accuracy-more than twice that of the next-best method under PGD attacks) and training efficiency (e.g., more than twice as fast as the second-best method in terms of accuracy), while maintaining superior standard accuracy in non-adversarial scenarios.



## **34. Mining Power Destruction Attacks in the Presence of Petty-Compliant Mining Pools**

cs.CR

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07410v1) [paper-pdf](http://arxiv.org/pdf/2502.07410v1)

**Authors**: Roozbeh Sarenche, Svetla Nikova, Bart Preneel

**Abstract**: Bitcoin's security relies on its Proof-of-Work consensus, where miners solve puzzles to propose blocks. The puzzle's difficulty is set by the difficulty adjustment mechanism (DAM), based on the network's available mining power. Attacks that destroy some portion of mining power can exploit the DAM to lower difficulty, making such attacks profitable. In this paper, we analyze three types of mining power destruction attacks in the presence of petty-compliant mining pools: selfish mining, bribery, and mining power distraction attacks. We analyze selfish mining while accounting for the distribution of mining power among pools, a factor often overlooked in the literature. Our findings indicate that selfish mining can be more destructive when the non-adversarial mining share is well distributed among pools. We also introduce a novel bribery attack, where the adversarial pool bribes petty-compliant pools to orphan others' blocks. For small pools, we demonstrate that the bribery attack can dominate strategies like selfish mining or undercutting. Lastly, we present the mining distraction attack, where the adversarial pool incentivizes petty-compliant pools to abandon Bitcoin's puzzle and mine for a simpler puzzle, thus wasting some part of their mining power. Similar to the previous attacks, this attack can lower the mining difficulty, but with the difference that it does not generate any evidence of mining power destruction, such as orphan blocks.



## **35. Enhancing Security and Privacy in Federated Learning using Low-Dimensional Update Representation and Proximity-Based Defense**

cs.CR

14 pages

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2405.18802v2) [paper-pdf](http://arxiv.org/pdf/2405.18802v2)

**Authors**: Wenjie Li, Kai Fan, Jingyuan Zhang, Hui Li, Wei Yang Bryan Lim, Qiang Yang

**Abstract**: Federated Learning (FL) is a promising privacy-preserving machine learning paradigm that allows data owners to collaboratively train models while keeping their data localized. Despite its potential, FL faces challenges related to the trustworthiness of both clients and servers, particularly against curious or malicious adversaries. In this paper, we introduce a novel framework named \underline{F}ederated \underline{L}earning with Low-Dimensional \underline{U}pdate \underline{R}epresentation and \underline{P}roximity-Based defense (FLURP), designed to address privacy preservation and resistance to Byzantine attacks in distributed learning environments. FLURP employs $\mathsf{LinfSample}$ method, enabling clients to compute the $l_{\infty}$ norm across sliding windows of updates, resulting in a Low-Dimensional Update Representation (LUR). Calculating the shared distance matrix among LURs, rather than updates, significantly reduces the overhead of Secure Multi-Party Computation (SMPC) by three orders of magnitude while effectively distinguishing between benign and poisoned updates. Additionally, FLURP integrates a privacy-preserving proximity-based defense mechanism utilizing optimized SMPC protocols to minimize communication rounds. Our experiments demonstrate FLURP's effectiveness in countering Byzantine adversaries with low communication and runtime overhead. FLURP offers a scalable framework for secure and reliable FL in distributed environments, facilitating its application in scenarios requiring robust data management and security.



## **36. CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models**

cs.CV

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07225v1) [paper-pdf](http://arxiv.org/pdf/2502.07225v1)

**Authors**: Sen Peng, Mingyue Wang, Jianfei He, Jijia Yang, Xiaohua Jia

**Abstract**: Latent diffusion models have recently demonstrated superior capabilities in many downstream image synthesis tasks. However, customization of latent diffusion models using unauthorized data can severely compromise the privacy and intellectual property rights of data owners. Adversarial examples as protective perturbations have been developed to defend against unauthorized data usage by introducing imperceptible noise to customization samples, preventing diffusion models from effectively learning them. In this paper, we first reveal that the primary reason adversarial examples are effective as protective perturbations in latent diffusion models is the distortion of their latent representations, as demonstrated through qualitative and quantitative experiments. We then propose the Contrastive Adversarial Training (CAT) utilizing adapters as an adaptive attack against these protection methods, highlighting their lack of robustness. Extensive experiments demonstrate that our CAT method significantly reduces the effectiveness of protective perturbations in customization configurations, urging the community to reconsider and enhance the robustness of existing protective perturbation methods. Code is available at \hyperlink{here}{https://github.com/senp98/CAT}.



## **37. LUNAR: LLM Unlearning via Neural Activation Redirection**

cs.LG

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07218v1) [paper-pdf](http://arxiv.org/pdf/2502.07218v1)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leaking private information. The ability to selectively remove knowledge from LLMs is, therefore, a highly desirable capability. In this paper, we propose LUNAR, a novel unlearning methodology grounded in the Linear Representation Hypothesis. LUNAR operates by redirecting the representations of unlearned data to regions that trigger the model's inherent ability to express its inability to answer. LUNAR achieves state-of-the-art unlearning performance while significantly enhancing the controllability of the unlearned model during inference. Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined "unlearning efficacy" and "model utility" score ("Deviation Score") on the PISTOL dataset across various base models. We also demonstrate, through quantitative analysis and qualitative examples, LUNAR's superior controllability in generating coherent and contextually aware responses, mitigating undesired side effects of existing methods. Moreover, we demonstrate that LUNAR is robust against white-box adversarial attacks and versatile in handling real-world scenarios, such as processing sequential unlearning requests.



## **38. SMAB: MAB based word Sensitivity Estimation Framework and its Applications in Adversarial Text Generation**

cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.07101v1) [paper-pdf](http://arxiv.org/pdf/2502.07101v1)

**Authors**: Saurabh Kumar Pandey, Sachin Vashistha, Debrup Das, Somak Aditya, Monojit Choudhury

**Abstract**: To understand the complexity of sequence classification tasks, Hahn et al. (2021) proposed sensitivity as the number of disjoint subsets of the input sequence that can each be individually changed to change the output. Though effective, calculating sensitivity at scale using this framework is costly because of exponential time complexity. Therefore, we introduce a Sensitivity-based Multi-Armed Bandit framework (SMAB), which provides a scalable approach for calculating word-level local (sentence-level) and global (aggregated) sensitivities concerning an underlying text classifier for any dataset. We establish the effectiveness of our approach through various applications. We perform a case study on CHECKLIST generated sentiment analysis dataset where we show that our algorithm indeed captures intuitively high and low-sensitive words. Through experiments on multiple tasks and languages, we show that sensitivity can serve as a proxy for accuracy in the absence of gold data. Lastly, we show that guiding perturbation prompts using sensitivity values in adversarial example generation improves attack success rate by 15.58%, whereas using sensitivity as an additional reward in adversarial paraphrase generation gives a 12.00% improvement over SOTA approaches. Warning: Contains potentially offensive content.



## **39. DROP: Poison Dilution via Knowledge Distillation for Federated Learning**

cs.LG

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.07011v1) [paper-pdf](http://arxiv.org/pdf/2502.07011v1)

**Authors**: Georgios Syros, Anshuman Suri, Farinaz Koushanfar, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Federated Learning is vulnerable to adversarial manipulation, where malicious clients can inject poisoned updates to influence the global model's behavior. While existing defense mechanisms have made notable progress, they fail to protect against adversaries that aim to induce targeted backdoors under different learning and attack configurations. To address this limitation, we introduce DROP (Distillation-based Reduction Of Poisoning), a novel defense mechanism that combines clustering and activity-tracking techniques with extraction of benign behavior from clients via knowledge distillation to tackle stealthy adversaries that manipulate low data poisoning rates and diverse malicious client ratios within the federation. Through extensive experimentation, our approach demonstrates superior robustness compared to existing defenses across a wide range of learning configurations. Finally, we evaluate existing defenses and our method under the challenging setting of non-IID client data distribution and highlight the challenges of designing a resilient FL defense in this setting.



## **40. Breaking Quantum Key Distributions under Quantum Switch-Based Attack**

quant-ph

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06780v1) [paper-pdf](http://arxiv.org/pdf/2502.06780v1)

**Authors**: Sumit Nandi, Biswaranjan Panda, Pankaj Agrawal, Arun K Pati

**Abstract**: Quantum key distribution (QKD) enables secure key sharing between distant parties, with several protocols proven resilient against conventional eavesdropping strategies. Here, we introduce a new attack scenario where an eavesdropper, Eve, exploits a quantum switch using the indefinite causal order to intercept and manipulate quantum communication channel. Using multiple metrics such as the information gain, mutual information, and Bell violation, we demonstrate that the presence of a quantum switch significantly compromises QKD security. Our results highlight a previously overlooked vulnerability, emphasizing the need for countermeasures against quantum-controlled adversarial strategies.



## **41. When Witnesses Defend: A Witness Graph Topological Layer for Adversarial Graph Learning**

cs.LG

Accepted at AAAI 2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2409.14161v3) [paper-pdf](http://arxiv.org/pdf/2409.14161v3)

**Authors**: Naheed Anjum Arafat, Debabrota Basu, Yulia Gel, Yuzhou Chen

**Abstract**: Capitalizing on the intuitive premise that shape characteristics are more robust to perturbations, we bridge adversarial graph learning with the emerging tools from computational topology, namely, persistent homology representations of graphs. We introduce the concept of witness complex to adversarial analysis on graphs, which allows us to focus only on the salient shape characteristics of graphs, yielded by the subset of the most essential nodes (i.e., landmarks), with minimal loss of topological information on the whole graph. The remaining nodes are then used as witnesses, governing which higher-order graph substructures are incorporated into the learning process. Armed with the witness mechanism, we design Witness Graph Topological Layer (WGTL), which systematically integrates both local and global topological graph feature representations, the impact of which is, in turn, automatically controlled by the robust regularized topological loss. Given the attacker's budget, we derive the important stability guarantees of both local and global topology encodings and the associated robust topological loss. We illustrate the versatility and efficiency of WGTL by its integration with five GNNs and three existing non-topological defense mechanisms. Our extensive experiments across six datasets demonstrate that WGTL boosts the robustness of GNNs across a range of perturbations and against a range of adversarial attacks. Our datasets and source codes are available at https://github.com/toggled/WGTL.



## **42. Tamper-Resistant Safeguards for Open-Weight LLMs**

cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.00761v4) [paper-pdf](http://arxiv.org/pdf/2408.00761v4)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after hundreds of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that progress on tamper-resistance is possible, opening up a promising new avenue to improve the safety and security of open-weight LLMs.



## **43. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Large Language Model(LLM) Based Emotion Inference Attacks**

cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18727v2) [paper-pdf](http://arxiv.org/pdf/2501.18727v2)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.



## **44. LIAR: Leveraging Inference Time Alignment (Best-of-N) to Jailbreak LLMs in Seconds**

cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2412.05232v2) [paper-pdf](http://arxiv.org/pdf/2412.05232v2)

**Authors**: James Beetham, Souradip Chakraborty, Mengdi Wang, Furong Huang, Amrit Singh Bedi, Mubarak Shah

**Abstract**: Traditional jailbreaks have successfully exposed vulnerabilities in LLMs, primarily relying on discrete combinatorial optimization, while more recent methods focus on training LLMs to generate adversarial prompts. However, both approaches are computationally expensive and slow, often requiring significant resources to generate a single successful attack. We hypothesize that the inefficiency of these methods arises from an inadequate characterization of the jailbreak problem itself. To address this gap, we approach the jailbreak problem as an alignment problem, leading us to propose LIAR (Leveraging Inference time Alignment to jailbReak), a fast and efficient best-of-N approach tailored for jailbreak attacks. LIAR offers several key advantages: it eliminates the need for additional training, operates in a fully black-box setting, significantly reduces computational overhead, and produces more human-readable adversarial prompts while maintaining competitive attack success rates. Our results demonstrate that a best-of-N approach is a simple yet highly effective strategy for evaluating the robustness of aligned LLMs, achieving attack success rates (ASR) comparable to state-of-the-art methods while offering a 10x improvement in perplexity and a significant speedup in Time-to-Attack, reducing execution time from tens of hours to seconds. Additionally, We also provide sub-optimality guarantees for the proposed LIAR. Our work highlights the potential of efficient, alignment-based jailbreak strategies for assessing and stress-testing AI safety measures.



## **45. Automatic ISA analysis for Secure Context Switching**

cs.OS

15 pages, 6 figures, 2 tables, 4 listings

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06609v1) [paper-pdf](http://arxiv.org/pdf/2502.06609v1)

**Authors**: Neelu S. Kalani, Thomas Bourgeat, Guerney D. H. Hunt, Wojciech Ozga

**Abstract**: Instruction set architectures are complex, with hundreds of registers and instructions that can modify dozens of them during execution, variably on each instance. Prose-style ISA specifications struggle to capture these intricacies of the ISAs, where often the important details about a single register are spread out across hundreds of pages of documentation. Ensuring that all ISA-state is swapped in context switch implementations of privileged software requires meticulous examination of these pages. This manual process is tedious and error-prone.   We propose a tool called Sailor that leverages machine-readable ISA specifications written in Sail to automate this task. Sailor determines the ISA-state necessary to swap during the context switch using the data collected from Sail and a novel algorithm to classify ISA-state as security-sensitive. Using Sailor's output, we identify three different classes of mishandled ISA-state across four open-source confidential computing systems. We further reveal five distinct security vulnerabilities that can be exploited using the mishandled ISA-state. This research exposes an often overlooked attack surface that stems from mishandled ISA-state, enabling unprivileged adversaries to exploit system vulnerabilities.



## **46. Krum Federated Chain (KFC): Using blockchain to defend against adversarial attacks in Federated Learning**

cs.LG

Submitted to Neural Networks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06917v1) [paper-pdf](http://arxiv.org/pdf/2502.06917v1)

**Authors**: Mario García-Márquez, Nuria Rodríguez-Barroso, M. Victoria Luzón, Francisco Herrera

**Abstract**: Federated Learning presents a nascent approach to machine learning, enabling collaborative model training across decentralized devices while safeguarding data privacy. However, its distributed nature renders it susceptible to adversarial attacks. Integrating blockchain technology with Federated Learning offers a promising avenue to enhance security and integrity. In this paper, we tackle the potential of blockchain in defending Federated Learning against adversarial attacks. First, we test Proof of Federated Learning, a well known consensus mechanism designed ad-hoc to federated contexts, as a defense mechanism demonstrating its efficacy against Byzantine and backdoor attacks when at least one miner remains uncompromised. Second, we propose Krum Federated Chain, a novel defense strategy combining Krum and Proof of Federated Learning, valid to defend against any configuration of Byzantine or backdoor attacks, even when all miners are compromised. Our experiments conducted on image classification datasets validate the effectiveness of our proposed approaches.



## **47. Robust Watermarks Leak: Channel-Aware Feature Extraction Enables Adversarial Watermark Manipulation**

cs.CV

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06418v1) [paper-pdf](http://arxiv.org/pdf/2502.06418v1)

**Authors**: Zhongjie Ba, Yitao Zhang, Peng Cheng, Bin Gong, Xinyu Zhang, Qinglong Wang, Kui Ren

**Abstract**: Watermarking plays a key role in the provenance and detection of AI-generated content. While existing methods prioritize robustness against real-world distortions (e.g., JPEG compression and noise addition), we reveal a fundamental tradeoff: such robust watermarks inherently improve the redundancy of detectable patterns encoded into images, creating exploitable information leakage. To leverage this, we propose an attack framework that extracts leakage of watermark patterns through multi-channel feature learning using a pre-trained vision model. Unlike prior works requiring massive data or detector access, our method achieves both forgery and detection evasion with a single watermarked image. Extensive experiments demonstrate that our method achieves a 60\% success rate gain in detection evasion and 51\% improvement in forgery accuracy compared to state-of-the-art methods while maintaining visual fidelity. Our work exposes the robustness-stealthiness paradox: current "robust" watermarks sacrifice security for distortion resistance, providing insights for future watermark design.



## **48. Amnesia as a Catalyst for Enhancing Black Box Pixel Attacks in Image Classification and Object Detection**

cs.CV

Accepted as a poster at NeurIPS 2024

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.07821v1) [paper-pdf](http://arxiv.org/pdf/2502.07821v1)

**Authors**: Dongsu Song, Daehwa Ko, Jay Hoon Jung

**Abstract**: It is well known that query-based attacks tend to have relatively higher success rates in adversarial black-box attacks. While research on black-box attacks is actively being conducted, relatively few studies have focused on pixel attacks that target only a limited number of pixels. In image classification, query-based pixel attacks often rely on patches, which heavily depend on randomness and neglect the fact that scattered pixels are more suitable for adversarial attacks. Moreover, to the best of our knowledge, query-based pixel attacks have not been explored in the field of object detection. To address these issues, we propose a novel pixel-based black-box attack called Remember and Forget Pixel Attack using Reinforcement Learning(RFPAR), consisting of two main components: the Remember and Forget processes. RFPAR mitigates randomness and avoids patch dependency by leveraging rewards generated through a one-step RL algorithm to perturb pixels. RFPAR effectively creates perturbed images that minimize the confidence scores while adhering to limited pixel constraints. Furthermore, we advance our proposed attack beyond image classification to object detection, where RFPAR reduces the confidence scores of detected objects to avoid detection. Experiments on the ImageNet-1K dataset for classification show that RFPAR outperformed state-of-the-art query-based pixel attacks. For object detection, using the MSCOCO dataset with YOLOv8 and DDQ, RFPAR demonstrates comparable mAP reduction to state-of-the-art query-based attack while requiring fewer query. Further experiments on the Argoverse dataset using YOLOv8 confirm that RFPAR effectively removed objects on a larger scale dataset. Our code is available at https://github.com/KAU-QuantumAILab/RFPAR.



## **49. Hyperparameters in Score-Based Membership Inference Attacks**

cs.LG

This work has been accepted for publication in the 3rd IEEE  Conference on Secure and Trustworthy Machine Learning (SaTML'25). The final  version will be available on IEEE Xplore

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06374v1) [paper-pdf](http://arxiv.org/pdf/2502.06374v1)

**Authors**: Gauri Pradhan, Joonas Jälkö, Marlon Tobaben, Antti Honkela

**Abstract**: Membership Inference Attacks (MIAs) have emerged as a valuable framework for evaluating privacy leakage by machine learning models. Score-based MIAs are distinguished, in particular, by their ability to exploit the confidence scores that the model generates for particular inputs. Existing score-based MIAs implicitly assume that the adversary has access to the target model's hyperparameters, which can be used to train the shadow models for the attack. In this work, we demonstrate that the knowledge of target hyperparameters is not a prerequisite for MIA in the transfer learning setting. Based on this, we propose a novel approach to select the hyperparameters for training the shadow models for MIA when the attacker has no prior knowledge about them by matching the output distributions of target and shadow models. We demonstrate that using the new approach yields hyperparameters that lead to an attack near indistinguishable in performance from an attack that uses target hyperparameters to train the shadow models. Furthermore, we study the empirical privacy risk of unaccounted use of training data for hyperparameter optimization (HPO) in differentially private (DP) transfer learning. We find no statistically significant evidence that performing HPO using training data would increase vulnerability to MIA.



## **50. POEX: Understanding and Mitigating Policy Executable Jailbreak Attacks against Embodied AI**

cs.RO

Homepage: https://poex-eai-jailbreak.github.io/

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2412.16633v2) [paper-pdf](http://arxiv.org/pdf/2412.16633v2)

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu

**Abstract**: Embodied AI systems are rapidly evolving due to the integration of LLMs as planning modules, which transform complex instructions into executable policies. However, LLMs are vulnerable to jailbreak attacks, which can generate malicious content. This paper investigates the feasibility and rationale behind applying traditional LLM jailbreak attacks to EAI systems. We aim to answer three questions: (1) Do traditional LLM jailbreak attacks apply to EAI systems? (2) What challenges arise if they do not? and (3) How can we defend against EAI jailbreak attacks? To this end, we first measure existing LLM-based EAI systems using a newly constructed dataset, i.e., the Harmful-RLbench. Our study confirms that traditional LLM jailbreak attacks are not directly applicable to EAI systems and identifies two unique challenges. First, the harmful text does not necessarily constitute harmful policies. Second, even if harmful policies can be generated, they are not necessarily executable by the EAI systems, which limits the potential risk. To facilitate a more comprehensive security analysis, we refine and introduce POEX, a novel red teaming framework that optimizes adversarial suffixes to induce harmful yet executable policies against EAI systems. The design of POEX employs adversarial constraints, policy evaluators, and suffix optimization to ensure successful policy execution while evading safety detection inside an EAI system. Experiments on the real-world robotic arm and simulator using Harmful-RLbench demonstrate the efficacy, highlighting severe safety vulnerabilities and high transferability across models. Finally, we propose prompt-based and model-based defenses, achieving an 85% success rate in mitigating attacks and enhancing safety awareness in EAI systems. Our findings underscore the urgent need for robust security measures to ensure the safe deployment of EAI in critical applications.



