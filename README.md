# Latest Adversarial Attack Papers
**update at 2024-01-01 16:28:26**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks**

cs.CV

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2310.06958v3) [paper-pdf](http://arxiv.org/pdf/2310.06958v3)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Ekaterina Shumitskaya, Sergey Lavrushkin, Dmitriy Vatolin

**Abstract**: Nowadays neural-network-based image- and video-quality metrics show better performance compared to traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. However, the adversarial robustness of image-quality metrics is also an area worth researching. In this paper, we analyse modern metrics' robustness to different adversarial attacks. We adopted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image/video-quality metrics. Some metrics showed high resistance to adversarial attacks which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts new metrics submissions for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. Try our benchmark using pip install robustness-benchmark.



## **2. Passive Inference Attacks on Split Learning via Adversarial Regularization**

cs.CR

17 pages, 20 pages

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2310.10483v2) [paper-pdf](http://arxiv.org/pdf/2310.10483v2)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.



## **3. MVPatch: More Vivid Patch for Adversarial Camouflaged Attacks on Object Detectors in the Physical World**

cs.CR

14 pages, 8 figures, submitted to IEEE Transactions on Information  Forensics & Security

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2312.17431v1) [paper-pdf](http://arxiv.org/pdf/2312.17431v1)

**Authors**: Zheng Zhou, Hongbo Zhao, Ju Liu, Qiaosheng Zhang, Guangbiao Wang, Chunlei Wang, Wenquan Feng

**Abstract**: Recent research has shown that adversarial patches can manipulate outputs from object detection models. However, the conspicuous patterns on these patches may draw more attention and raise suspicions among humans. Moreover, existing works have primarily focused on the attack performance of individual models and have neglected the generation of adversarial patches for ensemble attacks on multiple object detection models. To tackle these concerns, we propose a novel approach referred to as the More Vivid Patch (MVPatch), which aims to improve the transferability and stealthiness of adversarial patches while considering the limitations observed in prior paradigms, such as easy identification and poor transferability. Our approach incorporates an attack algorithm that decreases object confidence scores of multiple object detectors by using the ensemble attack loss function, thereby enhancing the transferability of adversarial patches. Additionally, we propose a lightweight visual similarity measurement algorithm realized by the Compared Specified Image Similarity (CSS) loss function, which allows for the generation of natural and stealthy adversarial patches without the reliance on additional generative models. Extensive experiments demonstrate that the proposed MVPatch algorithm achieves superior attack transferability compared to similar algorithms in both digital and physical domains, while also exhibiting a more natural appearance. These findings emphasize the remarkable stealthiness and transferability of the proposed MVPatch attack algorithm.



## **4. Can you See me? On the Visibility of NOPs against Android Malware Detectors**

cs.CR

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17356v1) [paper-pdf](http://arxiv.org/pdf/2312.17356v1)

**Authors**: Diego Soi, Davide Maiorca, Giorgio Giacinto, Harel Berger

**Abstract**: Android malware still represents the most significant threat to mobile systems. While Machine Learning systems are increasingly used to identify these threats, past studies have revealed that attackers can bypass these detection mechanisms by making subtle changes to Android applications, such as adding specific API calls. These modifications are often referred to as No OPerations (NOP), which ideally should not alter the semantics of the program. However, many NOPs can be spotted and eliminated by refining the app analysis process. This paper proposes a visibility metric that assesses the difficulty in spotting NOPs and similar non-operational codes. We tested our metric on a state-of-the-art, opcode-based deep learning system for Android malware detection. We implemented attacks on the feature and problem spaces and calculated their visibility according to our metric. The attained results show an intriguing trade-off between evasion efficacy and detectability: our metric can be valuable to ensure the real effectiveness of an adversarial attack, also serving as a useful aid to develop better defenses.



## **5. Timeliness: A New Design Metric and a New Attack Surface**

cs.IT

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17220v1) [paper-pdf](http://arxiv.org/pdf/2312.17220v1)

**Authors**: Priyanka Kaswan, Sennur Ulukus

**Abstract**: As the landscape of time-sensitive applications gains prominence in 5G/6G communications, timeliness of information updates at network nodes has become crucial, which is popularly quantified in the literature by the age of information metric. However, as we devise policies to improve age of information of our systems, we inadvertently introduce a new vulnerability for adversaries to exploit. In this article, we comprehensively discuss the diverse threats that age-based systems are vulnerable to. We begin with discussion on densely interconnected networks that employ gossiping between nodes to expedite dissemination of dynamic information in the network, and show how the age-based nature of gossiping renders these networks uniquely susceptible to threats such as timestomping attacks, jamming attacks, and the propagation of misinformation. Later, we survey adversarial works within simpler network settings, specifically in one-hop and two-hop configurations, and delve into adversarial robustness concerning challenges posed by jamming, timestomping, and issues related to privacy leakage. We conclude this article with future directions that aim to address challenges posed by more intelligent adversaries and robustness of networks to them.



## **6. Explainability-Based Adversarial Attack on Graphs Through Edge Perturbation**

cs.CR

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17301v1) [paper-pdf](http://arxiv.org/pdf/2312.17301v1)

**Authors**: Dibaloke Chanda, Saba Heidari Gheshlaghi, Nasim Yahya Soltani

**Abstract**: Despite the success of graph neural networks (GNNs) in various domains, they exhibit susceptibility to adversarial attacks. Understanding these vulnerabilities is crucial for developing robust and secure applications. In this paper, we investigate the impact of test time adversarial attacks through edge perturbations which involve both edge insertions and deletions. A novel explainability-based method is proposed to identify important nodes in the graph and perform edge perturbation between these nodes. The proposed method is tested for node classification with three different architectures and datasets. The results suggest that introducing edges between nodes of different classes has higher impact as compared to removing edges among nodes within the same class.



## **7. On the Robustness of Decision-Focused Learning**

cs.LG

17 pages, 45 figures, submitted to AAAI artificial intelligence for  operations research workshop

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2311.16487v3) [paper-pdf](http://arxiv.org/pdf/2311.16487v3)

**Authors**: Yehya Farhat

**Abstract**: Decision-Focused Learning (DFL) is an emerging learning paradigm that tackles the task of training a machine learning (ML) model to predict missing parameters of an incomplete optimization problem, where the missing parameters are predicted. DFL trains an ML model in an end-to-end system, by integrating the prediction and optimization tasks, providing better alignment of the training and testing objectives. DFL has shown a lot of promise and holds the capacity to revolutionize decision-making in many real-world applications. However, very little is known about the performance of these models under adversarial attacks. We adopt ten unique DFL methods and benchmark their performance under two distinctly focused attacks adapted towards the Predict-then-Optimize problem setting. Our study proposes the hypothesis that the robustness of a model is highly correlated with its ability to find predictions that lead to optimal decisions without deviating from the ground-truth label. Furthermore, we provide insight into how to target the models that violate this condition and show how these models respond differently depending on the achieved optimality at the end of their training cycles.



## **8. BlackboxBench: A Comprehensive Benchmark of Black-box Adversarial Attacks**

cs.CR

37 pages, 29 figures

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16979v1) [paper-pdf](http://arxiv.org/pdf/2312.16979v1)

**Authors**: Meixi Zheng, Xuanchen Yan, Zihao Zhu, Hongrui Chen, Baoyuan Wu

**Abstract**: Adversarial examples are well-known tools to evaluate the vulnerability of deep neural networks (DNNs). Although lots of adversarial attack algorithms have been developed, it is still challenging in the practical scenario that the model's parameters and architectures are inaccessible to the attacker/evaluator, i.e., black-box adversarial attacks. Due to the practical importance, there has been rapid progress from recent algorithms, reflected by the quick increase in attack success rate and the quick decrease in query numbers to the target model. However, there is a lack of thorough evaluations and comparisons among these algorithms, causing difficulties of tracking the real progress, analyzing advantages and disadvantages of different technical routes, as well as designing future development roadmap of this field. Thus, in this work, we aim at building a comprehensive benchmark of black-box adversarial attacks, called BlackboxBench. It mainly provides: 1) a unified, extensible and modular-based codebase, implementing 25 query-based attack algorithms and 30 transfer-based attack algorithms; 2) comprehensive evaluations: we evaluate the implemented algorithms against several mainstreaming model architectures on 2 widely used datasets (CIFAR-10 and a subset of ImageNet), leading to 14,106 evaluations in total; 3) thorough analysis and new insights, as well analytical tools. The website and source codes of BlackboxBench are available at https://blackboxbench.github.io/ and https://github.com/SCLBD/BlackboxBench/, respectively.



## **9. Attack Tree Analysis for Adversarial Evasion Attacks**

cs.CR

10 pages

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16957v1) [paper-pdf](http://arxiv.org/pdf/2312.16957v1)

**Authors**: Yuki Yamaguchi, Toshiaki Aoki

**Abstract**: Recently, the evolution of deep learning has promoted the application of machine learning (ML) to various systems. However, there are ML systems, such as autonomous vehicles, that cause critical damage when they misclassify. Conversely, there are ML-specific attacks called adversarial attacks based on the characteristics of ML systems. For example, one type of adversarial attack is an evasion attack, which uses minute perturbations called "adversarial examples" to intentionally misclassify classifiers. Therefore, it is necessary to analyze the risk of ML-specific attacks in introducing ML base systems. In this study, we propose a quantitative evaluation method for analyzing the risk of evasion attacks using attack trees. The proposed method consists of the extension of the conventional attack tree to analyze evasion attacks and the systematic construction method of the extension. In the extension of the conventional attack tree, we introduce ML and conventional attack nodes to represent various characteristics of evasion attacks. In the systematic construction process, we propose a procedure to construct the attack tree. The procedure consists of three steps: (1) organizing information about attack methods in the literature to a matrix, (2) identifying evasion attack scenarios from methods in the matrix, and (3) constructing the attack tree from the identified scenarios using a pattern. Finally, we conducted experiments on three ML image recognition systems to demonstrate the versatility and effectiveness of our proposed method.



## **10. DOEPatch: Dynamically Optimized Ensemble Model for Adversarial Patches Generation**

cs.CV

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16907v1) [paper-pdf](http://arxiv.org/pdf/2312.16907v1)

**Authors**: Wenyi Tan, Yang Li, Chenxing Zhao, Zhunga Liu, Quan Pan

**Abstract**: Object detection is a fundamental task in various applications ranging from autonomous driving to intelligent security systems. However, recognition of a person can be hindered when their clothing is decorated with carefully designed graffiti patterns, leading to the failure of object detection. To achieve greater attack potential against unknown black-box models, adversarial patches capable of affecting the outputs of multiple-object detection models are required. While ensemble models have proven effective, current research in the field of object detection typically focuses on the simple fusion of the outputs of all models, with limited attention being given to developing general adversarial patches that can function effectively in the physical world. In this paper, we introduce the concept of energy and treat the adversarial patches generation process as an optimization of the adversarial patches to minimize the total energy of the ``person'' category. Additionally, by adopting adversarial training, we construct a dynamically optimized ensemble model. During training, the weight parameters of the attacked target models are adjusted to find the balance point at which the generated adversarial patches can effectively attack all target models. We carried out six sets of comparative experiments and tested our algorithm on five mainstream object detection models. The adversarial patches generated by our algorithm can reduce the recognition accuracy of YOLOv2 and YOLOv3 to 13.19\% and 29.20\%, respectively. In addition, we conducted experiments to test the effectiveness of T-shirts covered with our adversarial patches in the physical world and could achieve that people are not recognized by the object detection model. Finally, leveraging the Grad-CAM tool, we explored the attack mechanism of adversarial patches from an energetic perspective.



## **11. Adversarial Attacks on Image Classification Models: Analysis and Defense**

cs.CV

This is the accepted version of the paper presented at the 10th  International Conference on Business Analytics and Intelligence (ICBAI'24).  The conference was organized by the Indian Institute of Science, Bangalore,  India, from December 18 - 20, 2023. The paper is 10 pages long and it  contains 14 tables and 11 figures

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16880v1) [paper-pdf](http://arxiv.org/pdf/2312.16880v1)

**Authors**: Jaydip Sen, Abhiraj Sen, Ananda Chatterjee

**Abstract**: The notion of adversarial attacks on image classification models based on convolutional neural networks (CNN) is introduced in this work. To classify images, deep learning models called CNNs are frequently used. However, when the networks are subject to adversarial attacks, extremely potent and previously trained CNN models that perform quite effectively on image datasets for image classification tasks may perform poorly. In this work, one well-known adversarial attack known as the fast gradient sign method (FGSM) is explored and its adverse effects on the performances of image classification models are examined. The FGSM attack is simulated on three pre-trained image classifier CNN architectures, ResNet-101, AlexNet, and RegNetY 400MF using randomly chosen images from the ImageNet dataset. The classification accuracies of the models are computed in the absence and presence of the attack to demonstrate the detrimental effect of the attack on the performances of the classifiers. Finally, a mechanism is proposed to defend against the FGSM attack based on a modified defensive distillation-based approach. Extensive results are presented for the validation of the proposed scheme.



## **12. Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model**

cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.11285v2) [paper-pdf](http://arxiv.org/pdf/2312.11285v2)

**Authors**: Decheng Liu, Xijun Wang, Chunlei Peng, Nannan Wang, Ruiming Hu, Xinbo Gao

**Abstract**: Adversarial attacks involve adding perturbations to the source image to cause misclassification by the target model, which demonstrates the potential of attacking face recognition models. Existing adversarial face image generation methods still can't achieve satisfactory performance because of low transferability and high detectability. In this paper, we propose a unified framework Adv-Diffusion that can generate imperceptible adversarial identity perturbations in the latent space but not the raw pixel space, which utilizes strong inpainting capabilities of the latent diffusion model to generate realistic adversarial images. Specifically, we propose the identity-sensitive conditioned diffusion generative model to generate semantic perturbations in the surroundings. The designed adaptive strength-based adversarial perturbation algorithm can ensure both attack transferability and stealthiness. Extensive qualitative and quantitative experiments on the public FFHQ and CelebA-HQ datasets prove the proposed method achieves superior performance compared with the state-of-the-art methods without an extra generative model training process. The source code is available at https://github.com/kopper-xdu/Adv-Diffusion.



## **13. Temporal Knowledge Distillation for Time-Sensitive Financial Services Applications**

cs.LG

arXiv admin note: text overlap with arXiv:2101.01689

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16799v1) [paper-pdf](http://arxiv.org/pdf/2312.16799v1)

**Authors**: Hongda Shen, Eren Kurshan

**Abstract**: Detecting anomalies has become an increasingly critical function in the financial service industry. Anomaly detection is frequently used in key compliance and risk functions such as financial crime detection fraud and cybersecurity. The dynamic nature of the underlying data patterns especially in adversarial environments like fraud detection poses serious challenges to the machine learning models. Keeping up with the rapid changes by retraining the models with the latest data patterns introduces pressures in balancing the historical and current patterns while managing the training data size. Furthermore the model retraining times raise problems in time-sensitive and high-volume deployment systems where the retraining period directly impacts the models ability to respond to ongoing attacks in a timely manner. In this study we propose a temporal knowledge distillation-based label augmentation approach (TKD) which utilizes the learning from older models to rapidly boost the latest model and effectively reduces the model retraining times to achieve improved agility. Experimental results show that the proposed approach provides advantages in retraining times while improving the model performance.



## **14. Multi-Task Models Adversarial Attacks**

cs.LG

19 pages, 6 figures

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2305.12066v3) [paper-pdf](http://arxiv.org/pdf/2305.12066v3)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Multi-Task Learning (MTL) involves developing a singular model, known as a multi-task model, to concurrently perform multiple tasks. While the security of single-task models has been thoroughly studied, multi-task models pose several critical security questions, such as 1) their vulnerability to single-task adversarial attacks, 2) the possibility of designing attacks that target multiple tasks, and 3) the impact of task sharing and adversarial training on their resilience to such attacks. This paper addresses these queries through detailed analysis and rigorous experimentation. First, we explore the adaptation of single-task white-box attacks to multi-task models and identify their limitations. We then introduce a novel attack framework, the Gradient Balancing Multi-Task Attack (GB-MTA), which treats attacking a multi-task model as an optimization problem. This problem, based on averaged relative loss change across tasks, is approximated as an integer linear programming problem. Extensive evaluations on MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrate GB-MTA's effectiveness against both standard and adversarially trained multi-task models. The results also highlight a trade-off between task accuracy improvement via parameter sharing and increased model vulnerability due to enhanced attack transferability.



## **15. Adversarial Attacks on LoRa Device Identification and Rogue Signal Detection with Deep Learning**

cs.CR

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16715v1) [paper-pdf](http://arxiv.org/pdf/2312.16715v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek

**Abstract**: Low-Power Wide-Area Network (LPWAN) technologies, such as LoRa, have gained significant attention for their ability to enable long-range, low-power communication for Internet of Things (IoT) applications. However, the security of LoRa networks remains a major concern, particularly in scenarios where device identification and classification of legitimate and spoofed signals are crucial. This paper studies a deep learning framework to address these challenges, considering LoRa device identification and legitimate vs. rogue LoRa device classification tasks. A deep neural network (DNN), either a convolutional neural network (CNN) or feedforward neural network (FNN), is trained for each task by utilizing real experimental I/Q data for LoRa signals, while rogue signals are generated by using kernel density estimation (KDE) of received signals by rogue devices. Fast Gradient Sign Method (FGSM)-based adversarial attacks are considered for LoRa signal classification tasks using deep learning models. The impact of these attacks is assessed on the performance of two tasks, namely device identification and legitimate vs. rogue device classification, by utilizing separate or common perturbations against these signal classification tasks. Results presented in this paper quantify the level of transferability of adversarial attacks on different LoRa signal classification tasks as a major vulnerability and highlight the need to make IoT applications robust to adversarial attacks.



## **16. Frauds Bargain Attack: Generating Adversarial Text Samples via Word Manipulation Process**

cs.CL

21 pages, 9 tables, 3 figures

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2303.01234v2) [paper-pdf](http://arxiv.org/pdf/2303.01234v2)

**Authors**: Mingze Ni, Zhensu Sun, Wei Liu

**Abstract**: Recent research has revealed that natural language processing (NLP) models are vulnerable to adversarial examples. However, the current techniques for generating such examples rely on deterministic heuristic rules, which fail to produce optimal adversarial examples. In response, this study proposes a new method called the Fraud's Bargain Attack (FBA), which uses a randomization mechanism to expand the search space and produce high-quality adversarial examples with a higher probability of success. FBA uses the Metropolis-Hasting sampler, a type of Markov Chain Monte Carlo sampler, to improve the selection of adversarial examples from all candidates generated by a customized stochastic process called the Word Manipulation Process (WMP). The WMP method modifies individual words in a contextually-aware manner through insertion, removal, or substitution. Through extensive experiments, this study demonstrates that FBA outperforms other methods in terms of attack success rate, imperceptibility and sentence quality.



## **17. Evaluating the security of CRYSTALS-Dilithium in the quantum random oracle model**

cs.CR

21 pages

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16619v1) [paper-pdf](http://arxiv.org/pdf/2312.16619v1)

**Authors**: Kelsey A. Jackson, Carl A. Miller, Daochen Wang

**Abstract**: In the wake of recent progress on quantum computing hardware, the National Institute of Standards and Technology (NIST) is standardizing cryptographic protocols that are resistant to attacks by quantum adversaries. The primary digital signature scheme that NIST has chosen is CRYSTALS-Dilithium. The hardness of this scheme is based on the hardness of three computational problems: Module Learning with Errors (MLWE), Module Short Integer Solution (MSIS), and SelfTargetMSIS. MLWE and MSIS have been well-studied and are widely believed to be secure. However, SelfTargetMSIS is novel and, though classically as hard as MSIS, its quantum hardness is unclear. In this paper, we provide the first proof of the hardness of SelfTargetMSIS via a reduction from MLWE in the Quantum Random Oracle Model (QROM). Our proof uses recently developed techniques in quantum reprogramming and rewinding. A central part of our approach is a proof that a certain hash function, derived from the MSIS problem, is collapsing. From this approach, we deduce a new security proof for Dilithium under appropriate parameter settings. Compared to the only other rigorous security proof for a variant of Dilithium, Dilithium-QROM, our proof has the advantage of being applicable under the condition q = 1 mod 2n, where q denotes the modulus and n the dimension of the underlying algebraic ring. This condition is part of the original Dilithium proposal and is crucial for the efficient implementation of the scheme. We provide new secure parameter sets for Dilithium under the condition q = 1 mod 2n, finding that our public key sizes and signature sizes are about 2.5 to 2.8 times larger than those of Dilithium-QROM for the same security levels.



## **18. Natural Adversarial Patch Generation Method Based on Latent Diffusion Model**

cs.CV

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16401v1) [paper-pdf](http://arxiv.org/pdf/2312.16401v1)

**Authors**: Xianyi Chen, Fazhan Liu, Dong Jiang, Kai Yan

**Abstract**: Recently, some research show that deep neural networks are vulnerable to the adversarial attacks, the well-trainned samples or patches could be used to trick the neural network detector or human visual perception. However, these adversarial patches, with their conspicuous and unusual patterns, lack camouflage and can easily raise suspicion in the real world. To solve this problem, this paper proposed a novel adversarial patch method called the Latent Diffusion Patch (LDP), in which, a pretrained encoder is first designed to compress the natural images into a feature space with key characteristics. Then trains the diffusion model using the above feature space. Finally, explore the latent space of the pretrained diffusion model using the image denoising technology. It polishes the patches and images through the powerful natural abilities of diffusion models, making them more acceptable to the human visual system. Experimental results, both digital and physical worlds, show that LDPs achieve a visual subjectivity score of 87.3%, while still maintaining effective attack capabilities.



## **19. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples**

cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.09520v2) [paper-pdf](http://arxiv.org/pdf/2312.09520v2)

**Authors**: Chen Ma, Ningfei Wang, Qi Alfred Chen, Chao Shen

**Abstract**: In Autonomous Driving (AD), real-time perception is a critical component responsible for detecting surrounding objects to ensure safe driving. While researchers have extensively explored the integrity of AD perception due to its safety and security implications, the aspect of availability (real-time performance) or latency has received limited attention. Existing works on latency-based attack have focused mainly on object detection, i.e., a component in camera-based AD perception, overlooking the entire camera-based AD perception, which hinders them to achieve effective system-level effects, such as vehicle crashes. In this paper, we propose SlowTrack, a novel framework for generating adversarial attacks to increase the execution time of camera-based AD perception. We propose a novel two-stage attack strategy along with the three new loss function designs. Our evaluation is conducted on four popular camera-based AD perception pipelines, and the results demonstrate that SlowTrack significantly outperforms existing latency-based attacks while maintaining comparable imperceptibility levels. Furthermore, we perform the evaluation on Baidu Apollo, an industry-grade full-stack AD system, and LGSVL, a production-grade AD simulator, with two scenarios to compare the system-level effects of SlowTrack and existing attacks. Our evaluation results show that the system-level effects can be significantly improved, i.e., the vehicle crash rate of SlowTrack is around 95% on average while existing works only have around 30%.



## **20. Model Stealing Attack against Recommender System**

cs.CR

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.11571v2) [paper-pdf](http://arxiv.org/pdf/2312.11571v2)

**Authors**: Zhihao Zhu, Rui Fan, Chenwang Wu, Yi Yang, Defu Lian, Enhong Chen

**Abstract**: Recent studies have demonstrated the vulnerability of recommender systems to data privacy attacks. However, research on the threat to model privacy in recommender systems, such as model stealing attacks, is still in its infancy. Some adversarial attacks have achieved model stealing attacks against recommender systems, to some extent, by collecting abundant training data of the target model (target data) or making a mass of queries. In this paper, we constrain the volume of available target data and queries and utilize auxiliary data, which shares the item set with the target data, to promote model stealing attacks. Although the target model treats target and auxiliary data differently, their similar behavior patterns allow them to be fused using an attention mechanism to assist attacks. Besides, we design stealing functions to effectively extract the recommendation list obtained by querying the target model. Experimental results show that the proposed methods are applicable to most recommender systems and various scenarios and exhibit excellent attack performance on multiple datasets.



## **21. MENLI: Robust Evaluation Metrics from Natural Language Inference**

cs.CL

TACL 2023 Camera-ready version; updated after proofreading by the  journal

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2208.07316v5) [paper-pdf](http://arxiv.org/pdf/2208.07316v5)

**Authors**: Yanran Chen, Steffen Eger

**Abstract**: Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).



## **22. Punctuation Matters! Stealthy Backdoor Attack for Language Models**

cs.CL

NLPCC 2023

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.15867v1) [paper-pdf](http://arxiv.org/pdf/2312.15867v1)

**Authors**: Xuan Sheng, Zhicheng Li, Zhaoyang Han, Xiangmao Chang, Piji Li

**Abstract**: Recent studies have pointed out that natural language processing (NLP) models are vulnerable to backdoor attacks. A backdoored model produces normal outputs on the clean samples while performing improperly on the texts with triggers that the adversary injects. However, previous studies on textual backdoor attack pay little attention to stealthiness. Moreover, some attack methods even cause grammatical issues or change the semantic meaning of the original texts. Therefore, they can easily be detected by humans or defense systems. In this paper, we propose a novel stealthy backdoor attack method against textual models, which is called \textbf{PuncAttack}. It leverages combinations of punctuation marks as the trigger and chooses proper locations strategically to replace them. Through extensive experiments, we demonstrate that the proposed method can effectively compromise multiple models in various tasks. Meanwhile, we conduct automatic evaluation and human inspection, which indicate the proposed method possesses good performance of stealthiness without bringing grammatical issues and altering the meaning of sentences.



## **23. Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**

cs.IR

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2312.15826v1) [paper-pdf](http://arxiv.org/pdf/2312.15826v1)

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Quoc Viet Hung Nguyen, Lizhen Cui, Hongzhi Yin

**Abstract**: Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.



## **24. Attention Deficit is Ordered! Fooling Deformable Vision Transformers with Collaborative Adversarial Patches**

cs.CV

12 pages, 14 figures

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2311.12914v2) [paper-pdf](http://arxiv.org/pdf/2311.12914v2)

**Authors**: Quazi Mishkatul Alam, Bilel Tarchoun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: The latest generation of transformer-based vision models has proven to be superior to Convolutional Neural Network (CNN)-based models across several vision tasks, largely attributed to their remarkable prowess in relation modeling. Deformable vision transformers significantly reduce the quadratic complexity of attention modeling by using sparse attention structures, enabling them to incorporate features across different scales and be used in large-scale applications, such as multi-view vision systems. Recent work has demonstrated adversarial attacks against conventional vision transformers; we show that these attacks do not transfer to deformable transformers due to their sparse attention structure. Specifically, attention in deformable transformers is modeled using pointers to the most relevant other tokens. In this work, we contribute for the first time adversarial attacks that manipulate the attention of deformable transformers, redirecting it to focus on irrelevant parts of the image. We also develop new collaborative attacks where a source patch manipulates attention to point to a target patch, which contains the adversarial noise to fool the model. In our experiments, we observe that altering less than 1% of the patched area in the input field results in a complete drop to 0% AP in single-view object detection using MS COCO and a 0% MODA in multi-view object detection using Wildtrack.



## **25. Adversarial Prompt Tuning for Vision-Language Models**

cs.CV

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2311.11261v2) [paper-pdf](http://arxiv.org/pdf/2311.11261v2)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code is available at https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.



## **26. Vulnerability of Machine Learning Approaches Applied in IoT-based Smart Grid: A Review**

cs.CR

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2308.15736v3) [paper-pdf](http://arxiv.org/pdf/2308.15736v3)

**Authors**: Zhenyong Zhang, Mengxiang Liu, Mingyang Sun, Ruilong Deng, Peng Cheng, Dusit Niyato, Mo-Yuen Chow, Jiming Chen

**Abstract**: Machine learning (ML) sees an increasing prevalence of being used in the internet-of-things (IoT)-based smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. We first highlight the specifics for constructing the adversarial attacks on MLsgAPPs. Then, the vulnerability of MLsgAPP is analyzed from both the aspects of the power system and ML model. Afterward, a comprehensive survey is conducted to review and compare existing studies about the adversarial attacks on MLsgAPPs in scenarios of generation, transmission, distribution, and consumption, and the countermeasures are reviewed according to the attacks that they defend against. Finally, the future research directions are discussed on the attacker's and defender's side, respectively. We also analyze the potential vulnerability of large language model-based (e.g., ChatGPT) power system applications. Overall, we encourage more researchers to contribute to investigating the adversarial issues of MLsgAPPs.



## **27. Privacy-Preserving Neural Graph Databases**

cs.DB

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2312.15591v1) [paper-pdf](http://arxiv.org/pdf/2312.15591v1)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Yangqiu Song

**Abstract**: In the era of big data and rapidly evolving information systems, efficient and accurate data retrieval has become increasingly crucial. Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (graph DBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data. The usage of neural embedding storage and complex neural logical query answering provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the database. Malicious attackers can infer more sensitive information in the database using well-designed combinatorial queries, such as by comparing the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training due to the privacy concerns. In this work, inspired by the privacy protection in graph embeddings, we propose a privacy-preserving neural graph database (P-NGDB) to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to force the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries. Extensive experiment results on three datasets show that P-NGDB can effectively protect private information in the graph database while delivering high-quality public answers responses to queries.



## **28. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

cs.SD

Accepted in ICASSP 2024

**SubmitDate**: 2023-12-24    [abs](http://arxiv.org/abs/2310.05354v3) [paper-pdf](http://arxiv.org/pdf/2310.05354v3)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.



## **29. Adversarial Data Poisoning for Fake News Detection: How to Make a Model Misclassify a Target News without Modifying It**

cs.LG

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.15228v1) [paper-pdf](http://arxiv.org/pdf/2312.15228v1)

**Authors**: Federico Siciliano, Luca Maiano, Lorenzo Papa, Federica Baccin, Irene Amerini, Fabrizio Silvestri

**Abstract**: Fake news detection models are critical to countering disinformation but can be manipulated through adversarial attacks. In this position paper, we analyze how an attacker can compromise the performance of an online learning detector on specific news content without being able to manipulate the original target news. In some contexts, such as social networks, where the attacker cannot exert complete control over all the information, this scenario can indeed be quite plausible. Therefore, we show how an attacker could potentially introduce poisoning data into the training data to manipulate the behavior of an online learning method. Our initial findings reveal varying susceptibility of logistic regression models based on complexity and attack type.



## **30. Towards Transferable Adversarial Attacks with Centralized Perturbation**

cs.CV

10 pages, 9 figures, accepted by AAAI 2024

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.06199v2) [paper-pdf](http://arxiv.org/pdf/2312.06199v2)

**Authors**: Shangbo Wu, Yu-an Tan, Yajie Wang, Ruinan Ma, Wencong Ma, Yuanzhang Li

**Abstract**: Adversarial transferability enables black-box attacks on unknown victim deep neural networks (DNNs), rendering attacks viable in real-world scenarios. Current transferable attacks create adversarial perturbation over the entire image, resulting in excessive noise that overfit the source model. Concentrating perturbation to dominant image regions that are model-agnostic is crucial to improving adversarial efficacy. However, limiting perturbation to local regions in the spatial domain proves inadequate in augmenting transferability. To this end, we propose a transferable adversarial attack with fine-grained perturbation optimization in the frequency domain, creating centralized perturbation. We devise a systematic pipeline to dynamically constrain perturbation optimization to dominant frequency coefficients. The constraint is optimized in parallel at each iteration, ensuring the directional alignment of perturbation optimization with model prediction. Our approach allows us to centralize perturbation towards sample-specific important frequency features, which are shared by DNNs, effectively mitigating source model overfitting. Experiments demonstrate that by dynamically centralizing perturbation on dominating frequency coefficients, crafted adversarial examples exhibit stronger transferability, and allowing them to bypass various defenses.



## **31. SODA: Protecting Proprietary Information in On-Device Machine Learning Models**

cs.LG

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.15036v1) [paper-pdf](http://arxiv.org/pdf/2312.15036v1)

**Authors**: Akanksha Atrey, Ritwik Sinha, Saayan Mitra, Prashant Shenoy

**Abstract**: The growth of low-end hardware has led to a proliferation of machine learning-based services in edge applications. These applications gather contextual information about users and provide some services, such as personalized offers, through a machine learning (ML) model. A growing practice has been to deploy such ML models on the user's device to reduce latency, maintain user privacy, and minimize continuous reliance on a centralized source. However, deploying ML models on the user's edge device can leak proprietary information about the service provider. In this work, we investigate on-device ML models that are used to provide mobile services and demonstrate how simple attacks can leak proprietary information of the service provider. We show that different adversaries can easily exploit such models to maximize their profit and accomplish content theft. Motivated by the need to thwart such attacks, we present an end-to-end framework, SODA, for deploying and serving on edge devices while defending against adversarial usage. Our results demonstrate that SODA can detect adversarial usage with 89% accuracy in less than 50 queries with minimal impact on service performance, latency, and storage.



## **32. Differentiable JPEG: The Devil is in the Details**

cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/ WACV paper:  https://openaccess.thecvf.com/content/WACV2024/html/Reich_Differentiable_JPEG_The_Devil_Is_in_the_Details_WACV_2024_paper.html

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2309.06978v4) [paper-pdf](http://arxiv.org/pdf/2309.06978v4)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.



## **33. Hierarchical Multi-Agent Reinforcement Learning for Assessing False-Data Injection Attacks on Transportation Networks**

cs.AI

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.14625v1) [paper-pdf](http://arxiv.org/pdf/2312.14625v1)

**Authors**: Taha Eghtesad, Sirui Li, Yevgeniy Vorobeychik, Aron Laszka

**Abstract**: The increasing reliance of drivers on navigation applications has made transportation networks more susceptible to data-manipulation attacks by malicious actors. Adversaries may exploit vulnerabilities in the data collection or processing of navigation services to inject false information, and to thus interfere with the drivers' route selection. Such attacks can significantly increase traffic congestions, resulting in substantial waste of time and resources, and may even disrupt essential services that rely on road networks. To assess the threat posed by such attacks, we introduce a computational framework to find worst-case data-injection attacks against transportation networks. First, we devise an adversarial model with a threat actor who can manipulate drivers by increasing the travel times that they perceive on certain roads. Then, we employ hierarchical multi-agent reinforcement learning to find an approximate optimal adversarial strategy for data manipulation. We demonstrate the applicability of our approach through simulating attacks on the Sioux Falls, ND network topology.



## **34. Complex Graph Laplacian Regularizer for Inferencing Grid States**

eess.SP

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2307.01906v2) [paper-pdf](http://arxiv.org/pdf/2307.01906v2)

**Authors**: Chinthaka Dinesh, Junfei Wang, Gene Cheung, Pirathayini Srikantha

**Abstract**: In order to maintain stable grid operations, system monitoring and control processes require the computation of grid states (e.g. voltage magnitude and angles) at high granularity. It is necessary to infer these grid states from measurements generated by a limited number of sensors like phasor measurement units (PMUs) that can be subjected to delays and losses due to channel artefacts, and/or adversarial attacks (e.g. denial of service, jamming, etc.). We propose a novel graph signal processing (GSP) based algorithm to interpolate states of the entire grid from observations of a small number of grid measurements. It is a two-stage process, where first an underlying Hermitian graph is learnt empirically from existing grid datasets. Then, the graph is used to interpolate missing grid signal samples in linear time. With our proposal, we can effectively reconstruct grid signals with significantly smaller number of observations when compared to existing traditional approaches (e.g. state estimation). In contrast to existing GSP approaches, we do not require knowledge of the underlying grid structure and parameters and are able to guarantee fast spectral optimization. We demonstrate the computational efficacy and accuracy of our proposal via practical studies conducted on the IEEE 118 bus system.



## **35. Backdoor Attack with Sparse and Invisible Trigger**

cs.CV

The first two authors contributed equally to this work. 13 pages

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2306.06209v2) [paper-pdf](http://arxiv.org/pdf/2306.06209v2)

**Authors**: Yinghua Gao, Yiming Li, Xueluan Gong, Zhifeng Li, Shu-Tao Xia, Qian Wang

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where the adversary manipulates a small portion of training data such that the victim model predicts normally on the benign samples but classifies the triggered samples as the target class. The backdoor attack is an emerging yet threatening training-phase threat, leading to serious risks in DNN-based applications. In this paper, we revisit the trigger patterns of existing backdoor attacks. We reveal that they are either visible or not sparse and therefore are not stealthy enough. More importantly, it is not feasible to simply combine existing methods to design an effective sparse and invisible backdoor attack. To address this problem, we formulate the trigger generation as a bi-level optimization problem with sparsity and invisibility constraints and propose an effective method to solve it. The proposed method is dubbed sparse and invisible backdoor attack (SIBA). We conduct extensive experiments on benchmark datasets under different settings, which verify the effectiveness of our attack and its resistance to existing backdoor defenses. The codes for reproducing main experiments are available at \url{https://github.com/YinghuaGao/SIBA}.



## **36. Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks**

cs.LG

preprint version

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.14440v1) [paper-pdf](http://arxiv.org/pdf/2312.14440v1)

**Authors**: Haz Sameen Shahgir, Xianghao Kong, Greg Ver Steeg, Yue Dong

**Abstract**: The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research into this, the reasons for their effectiveness are underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASRs). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix but is significantly harder in reverse. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We identify conditions resulting in a 60% success probability for adversarial attacks and others where this likelihood drops below 5%.



## **37. Elevating Defenses: Bridging Adversarial Training and Watermarking for Model Resilience**

cs.LG

Accepted at DAI Workshop, AAAI 2024

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14260v1) [paper-pdf](http://arxiv.org/pdf/2312.14260v1)

**Authors**: Janvi Thakkar, Giulio Zizzo, Sergio Maffeis

**Abstract**: Machine learning models are being used in an increasing number of critical applications; thus, securing their integrity and ownership is critical. Recent studies observed that adversarial training and watermarking have a conflicting interaction. This work introduces a novel framework to integrate adversarial training with watermarking techniques to fortify against evasion attacks and provide confident model verification in case of intellectual property theft. We use adversarial training together with adversarial watermarks to train a robust watermarked model. The key intuition is to use a higher perturbation budget to generate adversarial watermarks compared to the budget used for adversarial training, thus avoiding conflict. We use the MNIST and Fashion-MNIST datasets to evaluate our proposed technique on various model stealing attacks. The results obtained consistently outperform the existing baseline in terms of robustness performance and further prove the resilience of this defense against pruning and fine-tuning removal attacks.



## **38. Open-Set: ID Card Presentation Attack Detection using Neural Transfer Style**

cs.CV

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13993v1) [paper-pdf](http://arxiv.org/pdf/2312.13993v1)

**Authors**: Reuben Markham, Juan M. Espin, Mario Nieto-Hidalgo, Juan E. Tapia

**Abstract**: The accurate detection of ID card Presentation Attacks (PA) is becoming increasingly important due to the rising number of online/remote services that require the presentation of digital photographs of ID cards for digital onboarding or authentication. Furthermore, cybercriminals are continuously searching for innovative ways to fool authentication systems to gain unauthorized access to these services. Although advances in neural network design and training have pushed image classification to the state of the art, one of the main challenges faced by the development of fraud detection systems is the curation of representative datasets for training and evaluation. The handcrafted creation of representative presentation attack samples often requires expertise and is very time-consuming, thus an automatic process of obtaining high-quality data is highly desirable. This work explores ID card Presentation Attack Instruments (PAI) in order to improve the generation of samples with four Generative Adversarial Networks (GANs) based image translation models and analyses the effectiveness of the generated data for training fraud detection systems. Using open-source data, we show that synthetic attack presentations are an adequate complement for additional real attack presentations, where we obtain an EER performance increase of 0.63% points for print attacks and a loss of 0.29% for screen capture attacks.



## **39. AutoAugment Input Transformation for Highly Transferable Targeted Attacks**

cs.CV

10 pages, 6 figures

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14218v1) [paper-pdf](http://arxiv.org/pdf/2312.14218v1)

**Authors**: Haobo Lu, Xin Liu, Kun He

**Abstract**: Deep Neural Networks (DNNs) are widely acknowledged to be susceptible to adversarial examples, wherein imperceptible perturbations are added to clean examples through diverse input transformation attacks. However, these methods originally designed for non-targeted attacks exhibit low success rates in targeted attacks. Recent targeted adversarial attacks mainly pay attention to gradient optimization, attempting to find the suitable perturbation direction. However, few of them are dedicated to input transformation.In this work, we observe a positive correlation between the logit/probability of the target class and diverse input transformation methods in targeted attacks. To this end, we propose a novel targeted adversarial attack called AutoAugment Input Transformation (AAIT). Instead of relying on hand-made strategies, AAIT searches for the optimal transformation policy from a transformation space comprising various operations. Then, AAIT crafts adversarial examples using the found optimal transformation policy to boost the adversarial transferability in targeted attacks. Extensive experiments conducted on CIFAR-10 and ImageNet-Compatible datasets demonstrate that the proposed AAIT surpasses other transfer-based targeted attacks significantly.



## **40. Adversarial Infrared Curves: An Attack on Infrared Pedestrian Detectors in the Physical World**

cs.CR

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14217v1) [paper-pdf](http://arxiv.org/pdf/2312.14217v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Deep neural network security is a persistent concern, with considerable research on visible light physical attacks but limited exploration in the infrared domain. Existing approaches, like white-box infrared attacks using bulb boards and QR suits, lack realism and stealthiness. Meanwhile, black-box methods with cold and hot patches often struggle to ensure robustness. To bridge these gaps, we propose Adversarial Infrared Curves (AdvIC). Using Particle Swarm Optimization, we optimize two Bezier curves and employ cold patches in the physical realm to introduce perturbations, creating infrared curve patterns for physical sample generation. Our extensive experiments confirm AdvIC's effectiveness, achieving 94.8\% and 67.2\% attack success rates for digital and physical attacks, respectively. Stealthiness is demonstrated through a comparative analysis, and robustness assessments reveal AdvIC's superiority over baseline methods. When deployed against diverse advanced detectors, AdvIC achieves an average attack success rate of 76.8\%, emphasizing its robust nature. we explore adversarial defense strategies against AdvIC and examine its impact under various defense mechanisms. Given AdvIC's substantial security implications for real-world vision-based applications, urgent attention and mitigation efforts are warranted.



## **41. Quantum Neural Networks under Depolarization Noise: Exploring White-Box Attacks and Defenses**

quant-ph

Poster at Quantum Techniques in Machine Learning (QTML) 2023

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2311.17458v2) [paper-pdf](http://arxiv.org/pdf/2311.17458v2)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: Leveraging the unique properties of quantum mechanics, Quantum Machine Learning (QML) promises computational breakthroughs and enriched perspectives where traditional systems reach their boundaries. However, similarly to classical machine learning, QML is not immune to adversarial attacks. Quantum adversarial machine learning has become instrumental in highlighting the weak points of QML models when faced with adversarial crafted feature vectors. Diving deep into this domain, our exploration shines light on the interplay between depolarization noise and adversarial robustness. While previous results enhanced robustness from adversarial threats through depolarization noise, our findings paint a different picture. Interestingly, adding depolarization noise discontinued the effect of providing further robustness for a multi-class classification scenario. Consolidating our findings, we conducted experiments with a multi-class classifier adversarially trained on gate-based quantum simulators, further elucidating this unexpected behavior.



## **42. Where and How to Attack? A Causality-Inspired Recipe for Generating Counterfactual Adversarial Examples**

cs.LG

Accepted by AAAI-2024

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13628v1) [paper-pdf](http://arxiv.org/pdf/2312.13628v1)

**Authors**: Ruichu Cai, Yuxuan Zhu, Jie Qiao, Zefeng Liang, Furui Liu, Zhifeng Hao

**Abstract**: Deep neural networks (DNNs) have been demonstrated to be vulnerable to well-crafted \emph{adversarial examples}, which are generated through either well-conceived $\mathcal{L}_p$-norm restricted or unrestricted attacks. Nevertheless, the majority of those approaches assume that adversaries can modify any features as they wish, and neglect the causal generating process of the data, which is unreasonable and unpractical. For instance, a modification in income would inevitably impact features like the debt-to-income ratio within a banking system. By considering the underappreciated causal generating process, first, we pinpoint the source of the vulnerability of DNNs via the lens of causality, then give theoretical results to answer \emph{where to attack}. Second, considering the consequences of the attack interventions on the current state of the examples to generate more realistic adversarial examples, we propose CADE, a framework that can generate \textbf{C}ounterfactual \textbf{AD}versarial \textbf{E}xamples to answer \emph{how to attack}. The empirical results demonstrate CADE's effectiveness, as evidenced by its competitive performance across diverse attack scenarios, including white-box, transfer-based, and random intervention attacks.



## **43. ARBiBench: Benchmarking Adversarial Robustness of Binarized Neural Networks**

cs.CV

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13575v1) [paper-pdf](http://arxiv.org/pdf/2312.13575v1)

**Authors**: Peng Zhao, Jiehua Zhang, Bowen Peng, Longguang Wang, YingMei Wei, Yu Liu, Li Liu

**Abstract**: Network binarization exhibits great potential for deployment on resource-constrained devices due to its low computational cost. Despite the critical importance, the security of binarized neural networks (BNNs) is rarely investigated. In this paper, we present ARBiBench, a comprehensive benchmark to evaluate the robustness of BNNs against adversarial perturbations on CIFAR-10 and ImageNet. We first evaluate the robustness of seven influential BNNs on various white-box and black-box attacks. The results reveal that 1) The adversarial robustness of BNNs exhibits a completely opposite performance on the two datasets under white-box attacks. 2) BNNs consistently exhibit better adversarial robustness under black-box attacks. 3) Different BNNs exhibit certain similarities in their robustness performance. Then, we conduct experiments to analyze the adversarial robustness of BNNs based on these insights. Our research contributes to inspiring future research on enhancing the robustness of BNNs and advancing their application in real-world scenarios.



## **44. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

cs.CL

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14197v1) [paper-pdf](http://arxiv.org/pdf/2312.14197v1)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Keegan Hines, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: Recent remarkable advancements in large language models (LLMs) have led to their widespread adoption in various applications. A key feature of these applications is the combination of LLMs with external content, where user instructions and third-party content are combined to create prompts for LLM processing. These applications, however, are vulnerable to indirect prompt injection attacks, where malicious instructions embedded within external content compromise LLM's output, causing their responses to deviate from user expectations. Despite the discovery of this security issue, no comprehensive analysis of indirect prompt injection attacks on different LLMs is available due to the lack of a benchmark. Furthermore, no effective defense has been proposed.   In this work, we introduce the first benchmark, BIPIA, to measure the robustness of various LLMs and defenses against indirect prompt injection attacks. Our experiments reveal that LLMs with greater capabilities exhibit more vulnerable to indirect prompt injection attacks for text tasks, resulting in a higher ASR. We hypothesize that indirect prompt injection attacks are mainly due to the LLMs' inability to distinguish between instructions and external content. Based on this conjecture, we propose four black-box methods based on prompt learning and a white-box defense methods based on fine-tuning with adversarial training to enable LLMs to distinguish between instructions and external content and ignore instructions in the external content. Our experimental results show that our black-box defense methods can effectively reduce ASR but cannot completely thwart indirect prompt injection attacks, while our white-box defense method can reduce ASR to nearly zero with little adverse impact on the LLM's performance on general tasks. We hope that our benchmark and defenses can inspire future work in this important area.



## **45. Adversarial Purification with the Manifold Hypothesis**

cs.LG

Extended version of paper accepted at AAAI 2024 with supplementary  materials

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2210.14404v5) [paper-pdf](http://arxiv.org/pdf/2210.14404v5)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework for adversarial robustness using the manifold hypothesis. This framework provides sufficient conditions for defending against adversarial examples. We develop an adversarial purification method with this framework. Our method combines manifold learning with variational inference to provide adversarial robustness without the need for expensive adversarial training. Experimentally, our approach can provide adversarial robustness even if attackers are aware of the existence of the defense. In addition, our method can also serve as a test-time defense mechanism for variational autoencoders.



## **46. Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses**

cs.AI

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13435v1) [paper-pdf](http://arxiv.org/pdf/2312.13435v1)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world ML-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. The canonical approach in robustness evaluation calls for adaptive attacks, that is with complete knowledge of the defense and tailored to bypass it. In this study, we introduce a more expansive notion of being adaptive and show how attacks but also defenses can benefit by it and by learning from each other through interaction. We propose and evaluate a framework for adaptively optimizing black-box attacks and defenses against each other through the competitive game they form. To reliably measure robustness, it is important to evaluate against realistic and worst-case attacks. We thus augment both attacks and the evasive arsenal at their disposal through adaptive control, and observe that the same can be done for defenses, before we evaluate them first apart and then jointly under a multi-agent perspective. We demonstrate that active defenses, which control how the system responds, are a necessary complement to model hardening when facing decision-based attacks; then how these defenses can be circumvented by adaptive attacks, only to finally elicit active and adaptive defenses. We validate our observations through a wide theoretical and empirical investigation to confirm that AI-enabled adversaries pose a considerable threat to black-box ML-based systems, rekindling the proverbial arms race where defenses have to be AI-enabled too. Succinctly, we address the challenges posed by adaptive adversaries and develop adaptive defenses, thereby laying out effective strategies in ensuring the robustness of ML-based systems deployed in the real-world.



## **47. Universal and Transferable Adversarial Attacks on Aligned Language Models**

cs.CL

Website: http://llm-attacks.org/

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2307.15043v2) [paper-pdf](http://arxiv.org/pdf/2307.15043v2)

**Authors**: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson

**Abstract**: Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.



## **48. On the complexity of sabotage games for network security**

cs.CC

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13132v1) [paper-pdf](http://arxiv.org/pdf/2312.13132v1)

**Authors**: Dhananjay Raju, Georgios Bakirtzis, Ufuk Topcu

**Abstract**: Securing dynamic networks against adversarial actions is challenging because of the need to anticipate and counter strategic disruptions by adversarial entities within complex network structures. Traditional game-theoretic models, while insightful, often fail to model the unpredictability and constraints of real-world threat assessment scenarios. We refine sabotage games to reflect the realistic limitations of the saboteur and the network operator. By transforming sabotage games into reachability problems, our approach allows applying existing computational solutions to model realistic restrictions on attackers and defenders within the game. Modifying sabotage games into dynamic network security problems successfully captures the nuanced interplay of strategy and uncertainty in dynamic network security. Theoretically, we extend sabotage games to model network security contexts and thoroughly explore if the additional restrictions raise their computational complexity, often the bottleneck of game theory in practical contexts. Practically, this research sets the stage for actionable insights for developing robust defense mechanisms by understanding what risks to mitigate in dynamically changing networks under threat.



## **49. Prometheus: Infrastructure Security Posture Analysis with AI-generated Attack Graphs**

cs.CR

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13119v1) [paper-pdf](http://arxiv.org/pdf/2312.13119v1)

**Authors**: Xin Jin, Charalampos Katsis, Fan Sang, Jiahao Sun, Elisa Bertino, Ramana Rao Kompella, Ashish Kundu

**Abstract**: The rampant occurrence of cybersecurity breaches imposes substantial limitations on the progress of network infrastructures, leading to compromised data, financial losses, potential harm to individuals, and disruptions in essential services. The current security landscape demands the urgent development of a holistic security assessment solution that encompasses vulnerability analysis and investigates the potential exploitation of these vulnerabilities as attack paths. In this paper, we propose Prometheus, an advanced system designed to provide a detailed analysis of the security posture of computing infrastructures. Using user-provided information, such as device details and software versions, Prometheus performs a comprehensive security assessment. This assessment includes identifying associated vulnerabilities and constructing potential attack graphs that adversaries can exploit. Furthermore, Prometheus evaluates the exploitability of these attack paths and quantifies the overall security posture through a scoring mechanism. The system takes a holistic approach by analyzing security layers encompassing hardware, system, network, and cryptography. Furthermore, Prometheus delves into the interconnections between these layers, exploring how vulnerabilities in one layer can be leveraged to exploit vulnerabilities in others. In this paper, we present the end-to-end pipeline implemented in Prometheus, showcasing the systematic approach adopted for conducting this thorough security analysis.



## **50. LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate**

cs.LG

AAAI 2024

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13118v1) [paper-pdf](http://arxiv.org/pdf/2312.13118v1)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking \emph{given} pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose {\em Lipschitz Regularized Surrogate} (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.



