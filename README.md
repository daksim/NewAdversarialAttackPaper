# Latest Adversarial Attack Papers
**update at 2023-05-27 16:12:09**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. On the Robustness of Segment Anything**

cs.CV

22 pages

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.16220v1) [paper-pdf](http://arxiv.org/pdf/2305.16220v1)

**Authors**: Yihao Huang, Yue Cao, Tianlin Li, Felix Juefei-Xu, Di Lin, Ivor W. Tsang, Yang Liu, Qing Guo

**Abstract**: Segment anything model (SAM) has presented impressive objectness identification capability with the idea of prompt learning and a new collected large-scale dataset. Given a prompt (e.g., points, bounding boxes, or masks) and an input image, SAM is able to generate valid segment masks for all objects indicated by the prompts, presenting high generalization across diverse scenarios and being a general method for zero-shot transfer to downstream vision tasks. Nevertheless, it remains unclear whether SAM may introduce errors in certain threatening scenarios. Clarifying this is of significant importance for applications that require robustness, such as autonomous vehicles. In this paper, we aim to study the testing-time robustness of SAM under adversarial scenarios and common corruptions. To this end, we first build a testing-time robustness evaluation benchmark for SAM by integrating existing public datasets. Second, we extend representative adversarial attacks against SAM and study the influence of different prompts on robustness. Third, we study the robustness of SAM under diverse corruption types by evaluating SAM on corrupted datasets with different prompts. With experiments conducted on SA-1B and KITTI datasets, we find that SAM exhibits remarkable robustness against various corruptions, except for blur-related corruption. Furthermore, SAM remains susceptible to adversarial attacks, particularly when subjected to PGD and BIM attacks. We think such a comprehensive study could highlight the importance of the robustness issues of SAM and trigger a series of new tasks for SAM as well as downstream vision tasks.



## **2. ByzSecAgg: A Byzantine-Resistant Secure Aggregation Scheme for Federated Learning Based on Coded Computing and Vector Commitment**

cs.CR

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2302.09913v2) [paper-pdf](http://arxiv.org/pdf/2302.09913v2)

**Authors**: Tayyebeh Jahani-Nezhad, Mohammad Ali Maddah-Ali, Giuseppe Caire

**Abstract**: In this paper, we propose an efficient secure aggregation scheme for federated learning that is protected against Byzantine attacks and privacy leakages. Processing individual updates to manage adversarial behavior, while preserving privacy of data against colluding nodes, requires some sort of secure secret sharing. However, communication load for secret sharing of long vectors of updates can be very high. To resolve this issue, in the proposed scheme, local updates are partitioned into smaller sub-vectors and shared using ramp secret sharing. However, this sharing method does not admit bi-linear computations, such as pairwise distance calculations, needed by outlier-detection algorithms. To overcome this issue, each user runs another round of ramp sharing, with different embedding of data in the sharing polynomial. This technique, motivated by ideas from coded computing, enables secure computation of pairwise distance. In addition, to maintain the integrity and privacy of the local update, the proposed scheme also uses a vector commitment method, in which the commitment size remains constant (i.e. does not increase with the length of the local update), while simultaneously allowing verification of the secret sharing process.



## **3. Impact of Adversarial Training on Robustness and Generalizability of Language Models**

cs.CL

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2211.05523v2) [paper-pdf](http://arxiv.org/pdf/2211.05523v2)

**Authors**: Enes Altinisik, Hassan Sajjad, Husrev Taha Sencar, Safa Messaoud, Sanjay Chawla

**Abstract**: Adversarial training is widely acknowledged as the most effective defense against adversarial attacks. However, it is also well established that achieving both robustness and generalization in adversarially trained models involves a trade-off. The goal of this work is to provide an in depth comparison of different approaches for adversarial training in language models. Specifically, we study the effect of pre-training data augmentation as well as training time input perturbations vs. embedding space perturbations on the robustness and generalization of transformer-based language models. Our findings suggest that better robustness can be achieved by pre-training data augmentation or by training with input space perturbation. However, training with embedding space perturbation significantly improves generalization. A linguistic correlation analysis of neurons of the learned models reveals that the improved generalization is due to 'more specialized' neurons. To the best of our knowledge, this is the first work to carry out a deep qualitative analysis of different methods of generating adversarial examples in adversarial training of language models.



## **4. IDEA: Invariant Causal Defense for Graph Adversarial Robustness**

cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15792v1) [paper-pdf](http://arxiv.org/pdf/2305.15792v1)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Bingbing Xu, Xueqi Cheng

**Abstract**: Graph neural networks (GNNs) have achieved remarkable success in various tasks, however, their vulnerability to adversarial attacks raises concerns for the real-world applications. Existing defense methods can resist some attacks, but suffer unbearable performance degradation under other unknown attacks. This is due to their reliance on either limited observed adversarial examples to optimize (adversarial training) or specific heuristics to alter graph or model structures (graph purification or robust aggregation). In this paper, we propose an Invariant causal DEfense method against adversarial Attacks (IDEA), providing a new perspective to address this issue. The method aims to learn causal features that possess strong predictability for labels and invariant predictability across attacks, to achieve graph adversarial robustness. Through modeling and analyzing the causal relationships in graph adversarial attacks, we design two invariance objectives to learn the causal features. Extensive experiments demonstrate that our IDEA significantly outperforms all the baselines under both poisoning and evasion attacks on five benchmark datasets, highlighting the strong and invariant predictability of IDEA. The implementation of IDEA is available at https://anonymous.4open.science/r/IDEA_repo-666B.



## **5. Healing Unsafe Dialogue Responses with Weak Supervision Signals**

cs.CL

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15757v1) [paper-pdf](http://arxiv.org/pdf/2305.15757v1)

**Authors**: Zi Liang, Pinghui Wang, Ruofei Zhang, Shuo Zhang, Xiaofan Ye Yi Huang, Junlan Feng

**Abstract**: Recent years have seen increasing concerns about the unsafe response generation of large-scale dialogue systems, where agents will learn offensive or biased behaviors from the real-world corpus. Some methods are proposed to address the above issue by detecting and replacing unsafe training examples in a pipeline style. Though effective, they suffer from a high annotation cost and adapt poorly to unseen scenarios as well as adversarial attacks. Besides, the neglect of providing safe responses (e.g. simply replacing with templates) will cause the information-missing problem of dialogues. To address these issues, we propose an unsupervised pseudo-label sampling method, TEMP, that can automatically assign potential safe responses. Specifically, our TEMP method groups responses into several clusters and samples multiple labels with an adaptively sharpened sampling strategy, inspired by the observation that unsafe samples in the clusters are usually few and distribute in the tail. Extensive experiments in chitchat and task-oriented dialogues show that our TEMP outperforms state-of-the-art models with weak supervision signals and obtains comparable results under unsupervised learning settings.



## **6. PEARL: Preprocessing Enhanced Adversarial Robust Learning of Image Deraining for Semantic Segmentation**

cs.CV

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15709v1) [paper-pdf](http://arxiv.org/pdf/2305.15709v1)

**Authors**: Xianghao Jiao, Yaohua Liu, Jiaxin Gao, Xinyuan Chu, Risheng Liu, Xin Fan

**Abstract**: In light of the significant progress made in the development and application of semantic segmentation tasks, there has been increasing attention towards improving the robustness of segmentation models against natural degradation factors (e.g., rain streaks) or artificially attack factors (e.g., adversarial attack). Whereas, most existing methods are designed to address a single degradation factor and are tailored to specific application scenarios. In this work, we present the first attempt to improve the robustness of semantic segmentation tasks by simultaneously handling different types of degradation factors. Specifically, we introduce the Preprocessing Enhanced Adversarial Robust Learning (PEARL) framework based on the analysis of our proposed Naive Adversarial Training (NAT) framework. Our approach effectively handles both rain streaks and adversarial perturbation by transferring the robustness of the segmentation model to the image derain model. Furthermore, as opposed to the commonly used Negative Adversarial Attack (NAA), we design the Auxiliary Mirror Attack (AMA) to introduce positive information prior to the training of the PEARL framework, which improves defense capability and segmentation performance. Our extensive experiments and ablation studies based on different derain methods and segmentation models have demonstrated the significant performance improvement of PEARL with AMA in defense against various adversarial attacks and rain streaks while maintaining high generalization performance across different datasets.



## **7. Rethink Diversity in Deep Learning Testing**

cs.SE

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15698v1) [paper-pdf](http://arxiv.org/pdf/2305.15698v1)

**Authors**: Zi Wang, Jihye Choi, Somesh Jha

**Abstract**: Deep neural networks (DNNs) have demonstrated extraordinary capabilities and are an integral part of modern software systems. However, they also suffer from various vulnerabilities such as adversarial attacks and unfairness. Testing deep learning (DL) systems is therefore an important task, to detect and mitigate those vulnerabilities. Motivated by the success of traditional software testing, which often employs diversity heuristics, various diversity measures on DNNs have been proposed to help efficiently expose the buggy behavior of DNNs. In this work, we argue that many DNN testing tasks should be treated as directed testing problems rather than general-purpose testing tasks, because these tasks are specific and well-defined. Hence, the diversity-based approach is less effective.   Following our argument based on the semantics of DNNs and the testing goal, we derive $6$ metrics that can be used for DNN testing and carefully analyze their application scopes. We empirically show their efficacy in exposing bugs in DNNs compared to recent diversity-based metrics. Moreover, we also notice discrepancies between the practices of the software engineering (SE) community and the DL community. We point out some of these gaps, and hopefully, this can lead to bridging the SE practice and DL findings.



## **8. AdvFunMatch: When Consistent Teaching Meets Adversarial Robustness**

cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.14700v2) [paper-pdf](http://arxiv.org/pdf/2305.14700v2)

**Authors**: Zihui Wu, Haichang Gao, Bingqian Zhou, Ping Wang

**Abstract**: \emph{Consistent teaching} is an effective paradigm for implementing knowledge distillation (KD), where both student and teacher models receive identical inputs, and KD is treated as a function matching task (FunMatch). However, one limitation of FunMatch is that it does not account for the transfer of adversarial robustness, a model's resistance to adversarial attacks. To tackle this problem, we propose a simple but effective strategy called Adversarial Function Matching (AdvFunMatch), which aims to match distributions for all data points within the $\ell_p$-norm ball of the training data, in accordance with consistent teaching. Formulated as a min-max optimization problem, AdvFunMatch identifies the worst-case instances that maximizes the KL-divergence between teacher and student model outputs, which we refer to as "mismatched examples," and then matches the outputs on these mismatched examples. Our experimental results show that AdvFunMatch effectively produces student models with both high clean accuracy and robustness. Furthermore, we reveal that strong data augmentations (\emph{e.g.}, AutoAugment) are beneficial in AdvFunMatch, whereas prior works have found them less effective in adversarial training. Code is available at \url{https://gitee.com/zihui998/adv-fun-match}.



## **9. Near Optimal Adversarial Attack on UCB Bandits**

cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2008.09312v4) [paper-pdf](http://arxiv.org/pdf/2008.09312v4)

**Authors**: Shiliang Zuo

**Abstract**: I study a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. I propose a novel attack strategy that manipulates a learner employing the UCB algorithm into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\widehat{O}(\sqrt{\log T})$, where $T$ is the number of rounds. I also prove the first lower bound on the cumulative attack cost. The lower bound matches the upper bound up to $O(\log \log T)$ factors, showing the proposed attack strategy to be near optimal.



## **10. How do humans perceive adversarial text? A reality check on the validity and naturalness of word-based adversarial attacks**

cs.CL

ACL 2023

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15587v1) [paper-pdf](http://arxiv.org/pdf/2305.15587v1)

**Authors**: Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy

**Abstract**: Natural Language Processing (NLP) models based on Machine Learning (ML) are susceptible to adversarial attacks -- malicious algorithms that imperceptibly modify input text to force models into making incorrect predictions. However, evaluations of these attacks ignore the property of imperceptibility or study it under limited settings. This entails that adversarial perturbations would not pass any human quality gate and do not represent real threats to human-checked NLP systems. To bypass this limitation and enable proper assessment (and later, improvement) of NLP model robustness, we have surveyed 378 human participants about the perceptibility of text adversarial examples produced by state-of-the-art methods. Our results underline that existing text attacks are impractical in real-world scenarios where humans are involved. This contrasts with previous smaller-scale human studies, which reported overly optimistic conclusions regarding attack success. Through our work, we hope to position human perceptibility as a first-class success criterion for text attacks, and provide guidance for research to build effective attack algorithms and, in turn, design appropriate defence mechanisms.



## **11. Non-Asymptotic Lower Bounds For Training Data Reconstruction**

cs.LG

Additional experiments and minor bug fixes

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2303.16372v4) [paper-pdf](http://arxiv.org/pdf/2303.16372v4)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: Mathematical notions of privacy, such as differential privacy, are often stated as probabilistic guarantees that are difficult to interpret. It is imperative, however, that the implications of data sharing be effectively communicated to the data principal to ensure informed decision-making and offer full transparency with regards to the associated privacy risks. To this end, our work presents a rigorous quantitative evaluation of the protection conferred by private learners by investigating their resilience to training data reconstruction attacks. We accomplish this by deriving non-asymptotic lower bounds on the reconstruction error incurred by any adversary against $(\epsilon, \delta)$ differentially private learners for target samples that belong to any compact metric space. Working with a generalization of differential privacy, termed metric privacy, we remove boundedness assumptions on the input space prevalent in prior work, and prove that our results hold for general locally compact metric spaces. We extend the analysis to cover the high dimensional regime, wherein, the input data dimensionality may be larger than the adversary's query budget, and demonstrate that our bounds are minimax optimal under certain regimes.



## **12. Fast Adversarial CNN-based Perturbation Attack on No-Reference Image- and Video-Quality Metrics**

cs.CV

ICLR 2023 TinyPapers

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15544v1) [paper-pdf](http://arxiv.org/pdf/2305.15544v1)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Modern neural-network-based no-reference image- and video-quality metrics exhibit performance as high as full-reference metrics. These metrics are widely used to improve visual quality in computer vision methods and compare video processing methods. However, these metrics are not stable to traditional adversarial attacks, which can cause incorrect results. Our goal is to investigate the boundaries of no-reference metrics applicability, and in this paper, we propose a fast adversarial perturbation attack on no-reference quality metrics. The proposed attack (FACPA) can be exploited as a preprocessing step in real-time video processing and compression algorithms. This research can yield insights to further aid in designing of stable neural-network-based no-reference quality metrics.



## **13. Robust Classification via a Single Diffusion Model**

cs.CV

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15241v1) [paper-pdf](http://arxiv.org/pdf/2305.15241v1)

**Authors**: Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, Jun Zhu

**Abstract**: Recently, diffusion models have been successfully applied to improving adversarial robustness of image classifiers by purifying the adversarial noises or generating realistic data for adversarial training. However, the diffusion-based purification can be evaded by stronger adaptive attacks while adversarial training does not perform well under unseen threats, exhibiting inevitable limitations of these methods. To better harness the expressive power of diffusion models, in this paper we propose Robust Diffusion Classifier (RDC), a generative classifier that is constructed from a pre-trained diffusion model to be adversarially robust. Our method first maximizes the data likelihood of a given input and then predicts the class probabilities of the optimized input using the conditional likelihood of the diffusion model through Bayes' theorem. Since our method does not require training on particular adversarial attacks, we demonstrate that it is more generalizable to defend against multiple unseen threats. In particular, RDC achieves $73.24\%$ robust accuracy against $\ell_\infty$ norm-bounded perturbations with $\epsilon_\infty=8/255$ on CIFAR-10, surpassing the previous state-of-the-art adversarial training models by $+2.34\%$. The findings highlight the potential of generative classifiers by employing diffusion models for adversarial robustness compared with the commonly studied discriminative classifiers.



## **14. Adaptive Data Analysis in a Balanced Adversarial Model**

cs.LG

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15452v1) [paper-pdf](http://arxiv.org/pdf/2305.15452v1)

**Authors**: Kobbi Nissim, Uri Stemmer, Eliad Tsfadia

**Abstract**: In adaptive data analysis, a mechanism gets $n$ i.i.d. samples from an unknown distribution $D$, and is required to provide accurate estimations to a sequence of adaptively chosen statistical queries with respect to $D$. Hardt and Ullman (FOCS 2014) and Steinke and Ullman (COLT 2015) showed that in general, it is computationally hard to answer more than $\Theta(n^2)$ adaptive queries, assuming the existence of one-way functions.   However, these negative results strongly rely on an adversarial model that significantly advantages the adversarial analyst over the mechanism, as the analyst, who chooses the adaptive queries, also chooses the underlying distribution $D$. This imbalance raises questions with respect to the applicability of the obtained hardness results -- an analyst who has complete knowledge of the underlying distribution $D$ would have little need, if at all, to issue statistical queries to a mechanism which only holds a finite number of samples from $D$.   We consider more restricted adversaries, called \emph{balanced}, where each such adversary consists of two separated algorithms: The \emph{sampler} who is the entity that chooses the distribution and provides the samples to the mechanism, and the \emph{analyst} who chooses the adaptive queries, but does not have a prior knowledge of the underlying distribution. We improve the quality of previous lower bounds by revisiting them using an efficient \emph{balanced} adversary, under standard public-key cryptography assumptions. We show that these stronger hardness assumptions are unavoidable in the sense that any computationally bounded \emph{balanced} adversary that has the structure of all known attacks, implies the existence of public-key cryptography.



## **15. Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension**

cs.LG

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15203v1) [paper-pdf](http://arxiv.org/pdf/2305.15203v1)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto D'Onofrio, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification, neural networks are known to be vulnerable to adversarial attacks. These attacks are small perturbations of the input data designed to fool the model. Naturally, a question arises regarding the potential connection between the architecture, settings, or properties of the model and the nature of the attack. In this work, we aim to shed light on this problem by focusing on the implicit bias of the neural network, which refers to its inherent inclination to favor specific patterns or outcomes. Specifically, we investigate one aspect of the implicit bias, which involves the essential Fourier frequencies required for accurate image classification. We conduct tests to assess the statistical relationship between these frequencies and those necessary for a successful attack. To delve into this relationship, we propose a new method that can uncover non-linear correlations between sets of coordinates, which, in our case, are the aforementioned frequencies. By exploiting the entanglement between intrinsic dimension and correlation, we provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are closely tied.



## **16. IoT Threat Detection Testbed Using Generative Adversarial Networks**

cs.CR

8 pages, 5 figures

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15191v1) [paper-pdf](http://arxiv.org/pdf/2305.15191v1)

**Authors**: Farooq Shaikh, Elias Bou-Harb, Aldin Vehabovic, Jorge Crichigno, Aysegul Yayimli, Nasir Ghani

**Abstract**: The Internet of Things(IoT) paradigm provides persistent sensing and data collection capabilities and is becoming increasingly prevalent across many market sectors. However, most IoT devices emphasize usability and function over security, making them very vulnerable to malicious exploits. This concern is evidenced by the increased use of compromised IoT devices in large scale bot networks (botnets) to launch distributed denial of service(DDoS) attacks against high value targets. Unsecured IoT systems can also provide entry points to private networks, allowing adversaries relatively easy access to valuable resources and services. Indeed, these evolving IoT threat vectors (ranging from brute force attacks to remote code execution exploits) are posing key challenges. Moreover, many traditional security mechanisms are not amenable for deployment on smaller resource-constrained IoT platforms. As a result, researchers have been developing a range of methods for IoT security, with many strategies using advanced machine learning(ML) techniques. Along these lines, this paper presents a novel generative adversarial network(GAN) solution to detect threats from malicious IoT devices both inside and outside a network. This model is trained using both benign IoT traffic and global darknet data and further evaluated in a testbed with real IoT devices and malware threats.



## **17. Another Dead End for Morphological Tags? Perturbed Inputs and Parsing**

cs.CL

Accepted at Findings of ACL 2023

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15119v1) [paper-pdf](http://arxiv.org/pdf/2305.15119v1)

**Authors**: Alberto Muñoz-Ortiz, David Vilares

**Abstract**: The usefulness of part-of-speech tags for parsing has been heavily questioned due to the success of word-contextualized parsers. Yet, most studies are limited to coarse-grained tags and high quality written content; while we know little about their influence when it comes to models in production that face lexical errors. We expand these setups and design an adversarial attack to verify if the use of morphological information by parsers: (i) contributes to error propagation or (ii) if on the other hand it can play a role to correct mistakes that word-only neural parsers make. The results on 14 diverse UD treebanks show that under such attacks, for transition- and graph-based models their use contributes to degrade the performance even faster, while for the (lower-performing) sequence labeling parsers they are helpful. We also show that if morphological tags were utopically robust against lexical perturbations, they would be able to correct parsing mistakes.



## **18. Adversarial Demonstration Attacks on Large Language Models**

cs.CL

Work in Progress

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14950v1) [paper-pdf](http://arxiv.org/pdf/2305.14950v1)

**Authors**: Jiongxiao Wang, Zichen Liu, Keun Hee Park, Muhao Chen, Chaowei Xiao

**Abstract**: With the emergence of more powerful large language models (LLMs), such as ChatGPT and GPT-4, in-context learning (ICL) has gained significant prominence in leveraging these models for specific tasks by utilizing data-label pairs as precondition prompts. While incorporating demonstrations can greatly enhance the performance of LLMs across various tasks, it may introduce a new security concern: attackers can manipulate only the demonstrations without changing the input to perform an attack. In this paper, we investigate the security concern of ICL from an adversarial perspective, focusing on the impact of demonstrations. We propose an ICL attack based on TextAttack, which aims to only manipulate the demonstration without changing the input to mislead the models. Our results demonstrate that as the number of demonstrations increases, the robustness of in-context learning would decreases. Furthermore, we also observe that adversarially attacked demonstrations exhibit transferability to diverse input examples. These findings emphasize the critical security risks associated with ICL and underscore the necessity for extensive research on the robustness of ICL, particularly given its increasing significance in the advancement of LLMs.



## **19. Madvex: Instrumentation-based Adversarial Attacks on Machine Learning Malware Detection**

cs.CR

20 pages. To be published in The 20th Conference on Detection of  Intrusions and Malware & Vulnerability Assessment (DIMVA 2023)

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.02559v2) [paper-pdf](http://arxiv.org/pdf/2305.02559v2)

**Authors**: Nils Loose, Felix Mächtle, Claudius Pott, Volodymyr Bezsmertnyi, Thomas Eisenbarth

**Abstract**: WebAssembly (Wasm) is a low-level binary format for web applications, which has found widespread adoption due to its improved performance and compatibility with existing software. However, the popularity of Wasm has also led to its exploitation for malicious purposes, such as cryptojacking, where malicious actors use a victim's computing resources to mine cryptocurrencies without their consent. To counteract this threat, machine learning-based detection methods aiming to identify cryptojacking activities within Wasm code have emerged. It is well-known that neural networks are susceptible to adversarial attacks, where inputs to a classifier are perturbed with minimal changes that result in a crass misclassification. While applying changes in image classification is easy, manipulating binaries in an automated fashion to evade malware classification without changing functionality is non-trivial. In this work, we propose a new approach to include adversarial examples in the code section of binaries via instrumentation. The introduced gadgets allow for the inclusion of arbitrary bytes, enabling efficient adversarial attacks that reliably bypass state-of-the-art machine learning classifiers such as the CNN-based Minos recently proposed at NDSS 2021. We analyze the cost and reliability of instrumentation-based adversarial example generation and show that the approach works reliably at minimal size and performance overheads.



## **20. Introducing Competition to Boost the Transferability of Targeted Adversarial Examples through Clean Feature Mixup**

cs.CV

CVPR 2023 camera-ready

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14846v1) [paper-pdf](http://arxiv.org/pdf/2305.14846v1)

**Authors**: Junyoung Byun, Myung-Joon Kwon, Seungju Cho, Yoonji Kim, Changick Kim

**Abstract**: Deep neural networks are widely known to be susceptible to adversarial examples, which can cause incorrect predictions through subtle input modifications. These adversarial examples tend to be transferable between models, but targeted attacks still have lower attack success rates due to significant variations in decision boundaries. To enhance the transferability of targeted adversarial examples, we propose introducing competition into the optimization process. Our idea is to craft adversarial perturbations in the presence of two new types of competitor noises: adversarial perturbations towards different target classes and friendly perturbations towards the correct class. With these competitors, even if an adversarial example deceives a network to extract specific features leading to the target class, this disturbance can be suppressed by other competitors. Therefore, within this competition, adversarial examples should take different attack strategies by leveraging more diverse features to overwhelm their interference, leading to improving their transferability to different models. Considering the computational complexity, we efficiently simulate various interference from these two types of competitors in feature space by randomly mixing up stored clean features in the model inference and named this method Clean Feature Mixup (CFM). Our extensive experimental results on the ImageNet-Compatible and CIFAR-10 datasets show that the proposed method outperforms the existing baselines with a clear margin. Our code is available at https://github.com/dreamflake/CFM.



## **21. Block Coordinate Descent on Smooth Manifolds**

math.OC

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14744v1) [paper-pdf](http://arxiv.org/pdf/2305.14744v1)

**Authors**: Liangzu Peng, René Vidal

**Abstract**: Block coordinate descent is an optimization paradigm that iteratively updates one block of variables at a time, making it quite amenable to big data applications due to its scalability and performance. Its convergence behavior has been extensively studied in the (block-wise) convex case, but it is much less explored in the non-convex case. In this paper we analyze the convergence of block coordinate methods on non-convex sets and derive convergence rates on smooth manifolds under natural or weaker assumptions than prior work. Our analysis applies to many non-convex problems (e.g., generalized PCA, optimal transport, matrix factorization, Burer-Monteiro factorization, outlier-robust estimation, alternating projection, maximal coding rate reduction, neural collapse, adversarial attacks, homomorphic sensing), either yielding novel corollaries or recovering previously known results.



## **22. Adversarial Machine Learning and Cybersecurity: Risks, Challenges, and Legal Implications**

cs.CR

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.14553v1) [paper-pdf](http://arxiv.org/pdf/2305.14553v1)

**Authors**: Micah Musser, Andrew Lohn, James X. Dempsey, Jonathan Spring, Ram Shankar Siva Kumar, Brenda Leong, Christina Liaghati, Cindy Martinez, Crystal D. Grant, Daniel Rohrer, Heather Frase, Jonathan Elliott, John Bansemer, Mikel Rodriguez, Mitt Regan, Rumman Chowdhury, Stefan Hermanek

**Abstract**: In July 2022, the Center for Security and Emerging Technology (CSET) at Georgetown University and the Program on Geopolitics, Technology, and Governance at the Stanford Cyber Policy Center convened a workshop of experts to examine the relationship between vulnerabilities in artificial intelligence systems and more traditional types of software vulnerabilities. Topics discussed included the extent to which AI vulnerabilities can be handled under standard cybersecurity processes, the barriers currently preventing the accurate sharing of information about AI vulnerabilities, legal issues associated with adversarial attacks on AI systems, and potential areas where government support could improve AI vulnerability management and mitigation.   This report is meant to accomplish two things. First, it provides a high-level discussion of AI vulnerabilities, including the ways in which they are disanalogous to other types of vulnerabilities, and the current state of affairs regarding information sharing and legal oversight of AI vulnerabilities. Second, it attempts to articulate broad recommendations as endorsed by the majority of participants at the workshop.



## **23. Translate your gibberish: black-box adversarial attack on machine translation systems**

cs.CL

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2303.10974v2) [paper-pdf](http://arxiv.org/pdf/2303.10974v2)

**Authors**: Andrei Chertkov, Olga Tsymboi, Mikhail Pautov, Ivan Oseledets

**Abstract**: Neural networks are deployed widely in natural language processing tasks on the industrial scale, and perhaps the most often they are used as compounds of automatic machine translation systems. In this work, we present a simple approach to fool state-of-the-art machine translation tools in the task of translation from Russian to English and vice versa. Using a novel black-box gradient-free tensor-based optimizer, we show that many online translation tools, such as Google, DeepL, and Yandex, may both produce wrong or offensive translations for nonsensical adversarial input queries and refuse to translate seemingly benign input phrases. This vulnerability may interfere with understanding a new language and simply worsen the user's experience while using machine translation systems, and, hence, additional improvements of these tools are required to establish better translation.



## **24. The Best Defense is a Good Offense: Adversarial Augmentation against Adversarial Attacks**

cs.LG

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.14188v1) [paper-pdf](http://arxiv.org/pdf/2305.14188v1)

**Authors**: Iuri Frosio, Jan Kautz

**Abstract**: Many defenses against adversarial attacks (\eg robust classifiers, randomization, or image purification) use countermeasures put to work only after the attack has been crafted. We adopt a different perspective to introduce $A^5$ (Adversarial Augmentation Against Adversarial Attacks), a novel framework including the first certified preemptive defense against adversarial attacks. The main idea is to craft a defensive perturbation to guarantee that any attack (up to a given magnitude) towards the input in hand will fail. To this aim, we leverage existing automatic perturbation analysis tools for neural networks. We study the conditions to apply $A^5$ effectively, analyze the importance of the robustness of the to-be-defended classifier, and inspect the appearance of the robustified images. We show effective on-the-fly defensive augmentation with a robustifier network that ignores the ground truth label, and demonstrate the benefits of robustifier and classifier co-training. In our tests, $A^5$ consistently beats state of the art certified defenses on MNIST, CIFAR10, FashionMNIST and Tinyimagenet. We also show how to apply $A^5$ to create certifiably robust physical objects. Our code at https://github.com/NVlabs/A5 allows experimenting on a wide range of scenarios beyond the man-in-the-middle attack tested here, including the case of physical attacks.



## **25. Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing**

cs.LG

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2301.12554v2) [paper-pdf](http://arxiv.org/pdf/2301.12554v2)

**Authors**: Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi

**Abstract**: While prior research has proposed a plethora of methods that enhance the adversarial robustness of neural classifiers, practitioners are still reluctant to adopt these techniques due to their unacceptably severe penalties in clean accuracy. This paper shows that by mixing the output probabilities of a standard classifier and a robust model, where the standard network is optimized for clean accuracy and is not robust in general, this accuracy-robustness trade-off can be significantly alleviated. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key ingredient of this improvement. In addition to providing intuitive and empirical evidence, we also theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The proposed flexible method, termed "adaptive smoothing", can work in conjunction with existing or even future methods that improve clean accuracy, robustness, or adversary detection. Our empirical evaluation considers strong attack methods, including AutoAttack and adaptive attack. On the CIFAR-100 dataset, our method achieves an 85.21% clean accuracy while maintaining a 38.72% $\ell_\infty$-AutoAttacked ($\epsilon$=8/255) accuracy, becoming the second most robust method on the RobustBench CIFAR-100 benchmark as of submission, while improving the clean accuracy by ten percentage points compared with all listed models. The code that implements our method is available at https://github.com/Bai-YT/AdaptiveSmoothing.



## **26. Impact of Scaled Image on Robustness of Deep Neural Networks**

cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2209.02132v2) [paper-pdf](http://arxiv.org/pdf/2209.02132v2)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Deep neural networks (DNNs) have been widely used in computer vision tasks like image classification, object detection and segmentation. Whereas recent studies have shown their vulnerability to manual digital perturbations or distortion in the input images. The accuracy of the networks is remarkably influenced by the data distribution of their training dataset. Scaling the raw images creates out-of-distribution data, which makes it a possible adversarial attack to fool the networks. In this work, we propose a Scaling-distortion dataset ImageNet-CS by Scaling a subset of the ImageNet Challenge dataset by different multiples. The aim of our work is to study the impact of scaled images on the performance of advanced DNNs. We perform experiments on several state-of-the-art deep neural network architectures on the proposed ImageNet-CS, and the results show a significant positive correlation between scaling size and accuracy decline. Moreover, based on ResNet50 architecture, we demonstrate some tests on the performance of recent proposed robust training techniques and strategies like Augmix, Revisiting and Normalizer Free on our proposed ImageNet-CS. Experiment results have shown that these robust training techniques can improve networks' robustness to scaling transformation.



## **27. Impact of Colour Variation on Robustness of Deep Neural Networks**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.02132

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2209.02832v2) [paper-pdf](http://arxiv.org/pdf/2209.02832v2)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Deep neural networks (DNNs) have have shown state-of-the-art performance for computer vision applications like image classification, segmentation and object detection. Whereas recent advances have shown their vulnerability to manual digital perturbations in the input data, namely adversarial attacks. The accuracy of the networks is significantly affected by the data distribution of their training dataset. Distortions or perturbations on color space of input images generates out-of-distribution data, which make networks more likely to misclassify them. In this work, we propose a color-variation dataset by distorting their RGB color on a subset of the ImageNet with 27 different combinations. The aim of our work is to study the impact of color variation on the performance of DNNs. We perform experiments on several state-of-the-art DNN architectures on the proposed dataset, and the result shows a significant correlation between color variation and loss of accuracy. Furthermore, based on the ResNet50 architecture, we demonstrate some experiments of the performance of recently proposed robust training techniques and strategies, such as Augmix, revisit, and free normalizer, on our proposed dataset. Experimental results indicate that these robust training techniques can improve the robustness of deep networks to color variation.



## **28. Adversarial Zoom Lens: A Novel Physical-World Attack to DNNs**

cs.CR

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2206.12251v2) [paper-pdf](http://arxiv.org/pdf/2206.12251v2)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Although deep neural networks (DNNs) are known to be fragile, no one has studied the effects of zooming-in and zooming-out of images in the physical world on DNNs performance. In this paper, we demonstrate a novel physical adversarial attack technique called Adversarial Zoom Lens (AdvZL), which uses a zoom lens to zoom in and out of pictures of the physical world, fooling DNNs without changing the characteristics of the target object. The proposed method is so far the only adversarial attack technique that does not add physical adversarial perturbation attack DNNs. In a digital environment, we construct a data set based on AdvZL to verify the antagonism of equal-scale enlarged images to DNNs. In the physical environment, we manipulate the zoom lens to zoom in and out of the target object, and generate adversarial samples. The experimental results demonstrate the effectiveness of AdvZL in both digital and physical environments. We further analyze the antagonism of the proposed data set to the improved DNNs. On the other hand, we provide a guideline for defense against AdvZL by means of adversarial training. Finally, we look into the threat possibilities of the proposed approach to future autonomous driving and variant attack ideas similar to the proposed attack.



## **29. Impact of Light and Shadow on Robustness of Deep Neural Networks**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.02832,  arXiv:2209.02132

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.14165v1) [paper-pdf](http://arxiv.org/pdf/2305.14165v1)

**Authors**: Chengyin Hu, Weiwen Shi, Chao Li, Jialiang Sun, Donghua Wang, Junqi Wu, Guijian Tang

**Abstract**: Deep neural networks (DNNs) have made remarkable strides in various computer vision tasks, including image classification, segmentation, and object detection. However, recent research has revealed a vulnerability in advanced DNNs when faced with deliberate manipulations of input data, known as adversarial attacks. Moreover, the accuracy of DNNs is heavily influenced by the distribution of the training dataset. Distortions or perturbations in the color space of input images can introduce out-of-distribution data, resulting in misclassification. In this work, we propose a brightness-variation dataset, which incorporates 24 distinct brightness levels for each image within a subset of ImageNet. This dataset enables us to simulate the effects of light and shadow on the images, so as is to investigate the impact of light and shadow on the performance of DNNs. In our study, we conduct experiments using several state-of-the-art DNN architectures on the aforementioned dataset. Through our analysis, we discover a noteworthy positive correlation between the brightness levels and the loss of accuracy in DNNs. Furthermore, we assess the effectiveness of recently proposed robust training techniques and strategies, including AugMix, Revisit, and Free Normalizer, using the ResNet50 architecture on our brightness-variation dataset. Our experimental results demonstrate that these techniques can enhance the robustness of DNNs against brightness variation, leading to improved performance when dealing with images exhibiting varying brightness levels.



## **30. QFA2SR: Query-Free Adversarial Transfer Attacks to Speaker Recognition Systems**

cs.CR

Accepted by the 32nd USENIX Security Symposium (2023 USENIX  Security); Full Version

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.14097v1) [paper-pdf](http://arxiv.org/pdf/2305.14097v1)

**Authors**: Guangke Chen, Yedi Zhang, Zhe Zhao, Fu Song

**Abstract**: Current adversarial attacks against speaker recognition systems (SRSs) require either white-box access or heavy black-box queries to the target SRS, thus still falling behind practical attacks against proprietary commercial APIs and voice-controlled devices. To fill this gap, we propose QFA2SR, an effective and imperceptible query-free black-box attack, by leveraging the transferability of adversarial voices. To improve transferability, we present three novel methods, tailored loss functions, SRS ensemble, and time-freq corrosion. The first one tailors loss functions to different attack scenarios. The latter two augment surrogate SRSs in two different ways. SRS ensemble combines diverse surrogate SRSs with new strategies, amenable to the unique scoring characteristics of SRSs. Time-freq corrosion augments surrogate SRSs by incorporating well-designed time-/frequency-domain modification functions, which simulate and approximate the decision boundary of the target SRS and distortions introduced during over-the-air attacks. QFA2SR boosts the targeted transferability by 20.9%-70.7% on four popular commercial APIs (Microsoft Azure, iFlytek, Jingdong, and TalentedSoft), significantly outperforming existing attacks in query-free setting, with negligible effect on the imperceptibility. QFA2SR is also highly effective when launched over the air against three wide-spread voice assistants (Google Assistant, Apple Siri, and TMall Genie) with 60%, 46%, and 70% targeted transferability, respectively.



## **31. Adversarial Catoptric Light: An Effective, Stealthy and Robust Physical-World Attack to DNNs**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.09652,  arXiv:2209.02430

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2209.11739v2) [paper-pdf](http://arxiv.org/pdf/2209.11739v2)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Deep neural networks (DNNs) have demonstrated exceptional success across various tasks, underscoring the need to evaluate the robustness of advanced DNNs. However, traditional methods using stickers as physical perturbations to deceive classifiers present challenges in achieving stealthiness and suffer from printing loss. Recent advancements in physical attacks have utilized light beams such as lasers and projectors to perform attacks, where the optical patterns generated are artificial rather than natural. In this study, we introduce a novel physical attack, adversarial catoptric light (AdvCL), where adversarial perturbations are generated using a common natural phenomenon, catoptric light, to achieve stealthy and naturalistic adversarial attacks against advanced DNNs in a black-box setting. We evaluate the proposed method in three aspects: effectiveness, stealthiness, and robustness. Quantitative results obtained in simulated environments demonstrate the effectiveness of the proposed method, and in physical scenarios, we achieve an attack success rate of 83.5%, surpassing the baseline. We use common catoptric light as a perturbation to enhance the stealthiness of the method and make physical samples appear more natural. Robustness is validated by successfully attacking advanced and robust DNNs with a success rate over 80% in all cases. Additionally, we discuss defense strategy against AdvCL and put forward some light-based physical attacks.



## **32. MultiRobustBench: Benchmarking Robustness Against Multiple Attacks**

cs.LG

ICML 2023

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2302.10980v2) [paper-pdf](http://arxiv.org/pdf/2302.10980v2)

**Authors**: Sihui Dai, Saeed Mahloujifar, Chong Xiang, Vikash Sehwag, Pin-Yu Chen, Prateek Mittal

**Abstract**: The bulk of existing research in defending against adversarial examples focuses on defending against a single (typically bounded Lp-norm) attack, but for a practical setting, machine learning (ML) models should be robust to a wide variety of attacks. In this paper, we present the first unified framework for considering multiple attacks against ML models. Our framework is able to model different levels of learner's knowledge about the test-time adversary, allowing us to model robustness against unforeseen attacks and robustness against unions of attacks. Using our framework, we present the first leaderboard, MultiRobustBench, for benchmarking multiattack evaluation which captures performance across attack types and attack strengths. We evaluate the performance of 16 defended models for robustness against a set of 9 different attack types, including Lp-based threat models, spatial transformations, and color changes, at 20 different attack strengths (180 attacks total). Additionally, we analyze the state of current defenses against multiple attacks. Our analysis shows that while existing defenses have made progress in terms of average robustness across the set of attacks used, robustness against the worst-case attack is still a big open problem as all existing models perform worse than random guessing.



## **33. Adversarial Color Film: Effective Physical-World Attack to DNNs**

cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2209.02430v2) [paper-pdf](http://arxiv.org/pdf/2209.02430v2)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: It is well known that the performance of deep neural networks (DNNs) is susceptible to subtle interference. So far, camera-based physical adversarial attacks haven't gotten much attention, but it is the vacancy of physical attack. In this paper, we propose a simple and efficient camera-based physical attack called Adversarial Color Film (AdvCF), which manipulates the physical parameters of color film to perform attacks. Carefully designed experiments show the effectiveness of the proposed method in both digital and physical environments. In addition, experimental results show that the adversarial samples generated by AdvCF have excellent performance in attack transferability, which enables AdvCF effective black-box attacks. At the same time, we give the guidance of defense against AdvCF by means of adversarial training. Finally, we look into AdvCF's threat to future vision-based systems and propose some promising mentality for camera-based physical attacks.



## **34. Expressive Losses for Verified Robustness via Convex Combinations**

cs.LG

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13991v1) [paper-pdf](http://arxiv.org/pdf/2305.13991v1)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth, Alessio Lomuscio

**Abstract**: In order to train networks for verified adversarial robustness, previous work typically over-approximates the worst-case loss over (subsets of) perturbation regions or induces verifiability on top of adversarial training. The key to state-of-the-art performance lies in the expressivity of the employed loss function, which should be able to match the tightness of the verifiers to be employed post-training. We formalize a definition of expressivity, and show that it can be satisfied via simple convex combinations between adversarial attacks and IBP bounds. We then show that the resulting algorithms, named CC-IBP and MTL-IBP, yield state-of-the-art results across a variety of settings in spite of their conceptual simplicity. In particular, for $\ell_\infty$ perturbations of radius $\frac{1}{255}$ on TinyImageNet and downscaled ImageNet, MTL-IBP improves on the best standard and verified accuracies from the literature by from $1.98\%$ to $3.92\%$ points while only relying on single-step adversarial attacks.



## **35. Adversarial Color Projection: A Projector-based Physical Attack to DNNs**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2209.02430

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2209.09652v2) [paper-pdf](http://arxiv.org/pdf/2209.09652v2)

**Authors**: Chengyin Hu, Weiwen Shi, Ling Tian

**Abstract**: Recent research has demonstrated that deep neural networks (DNNs) are vulnerable to adversarial perturbations. Therefore, it is imperative to evaluate the resilience of advanced DNNs to adversarial attacks. However, traditional methods that use stickers as physical perturbations to deceive classifiers face challenges in achieving stealthiness and are susceptible to printing loss. Recently, advancements in physical attacks have utilized light beams, such as lasers, to perform attacks, where the optical patterns generated are artificial rather than natural. In this work, we propose a black-box projector-based physical attack, referred to as adversarial color projection (AdvCP), which manipulates the physical parameters of color projection to perform an adversarial attack. We evaluate our approach on three crucial criteria: effectiveness, stealthiness, and robustness. In the digital environment, we achieve an attack success rate of 97.60% on a subset of ImageNet, while in the physical environment, we attain an attack success rate of 100% in the indoor test and 82.14% in the outdoor test. The adversarial samples generated by AdvCP are compared with baseline samples to demonstrate the stealthiness of our approach. When attacking advanced DNNs, experimental results show that our method can achieve more than 85% attack success rate in all cases, which verifies the robustness of AdvCP. Finally, we consider the potential threats posed by AdvCP to future vision-based systems and applications and suggest some ideas for light-based physical attacks.



## **36. Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models**

cs.CV

To Appear in the ACM Conference on Computer and Communications  Security, November 26, 2023

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13873v1) [paper-pdf](http://arxiv.org/pdf/2305.13873v1)

**Authors**: Yiting Qu, Xinyue Shen, Xinlei He, Michael Backes, Savvas Zannettou, Yang Zhang

**Abstract**: State-of-the-art Text-to-Image models like Stable Diffusion and DALLE$\cdot$2 are revolutionizing how people generate visual content. At the same time, society has serious concerns about how adversaries can exploit such models to generate unsafe images. In this work, we focus on demystifying the generation of unsafe images and hateful memes from Text-to-Image models. We first construct a typology of unsafe images consisting of five categories (sexually explicit, violent, disturbing, hateful, and political). Then, we assess the proportion of unsafe images generated by four advanced Text-to-Image models using four prompt datasets. We find that these models can generate a substantial percentage of unsafe images; across four models and four prompt datasets, 14.56% of all generated images are unsafe. When comparing the four models, we find different risk levels, with Stable Diffusion being the most prone to generating unsafe content (18.92% of all generated images are unsafe). Given Stable Diffusion's tendency to generate more unsafe content, we evaluate its potential to generate hateful meme variants if exploited by an adversary to attack a specific individual or community. We employ three image editing methods, DreamBooth, Textual Inversion, and SDEdit, which are supported by Stable Diffusion. Our evaluation result shows that 24% of the generated images using DreamBooth are hateful meme variants that present the features of the original hateful meme and the target individual/community; these generated images are comparable to hateful meme variants collected from the real world. Overall, our results demonstrate that the danger of large-scale generation of unsafe images is imminent. We discuss several mitigating measures, such as curating training data, regulating prompts, and implementing safety filters, and encourage better safeguard tools to be developed to prevent unsafe generation.



## **37. A Study on the Efficiency and Generalization of Light Hybrid Retrievers**

cs.IR

accepted to ACL23

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2210.01371v2) [paper-pdf](http://arxiv.org/pdf/2210.01371v2)

**Authors**: Man Luo, Shashank Jain, Anchit Gupta, Arash Einolghozati, Barlas Oguz, Debojeet Chatterjee, Xilun Chen, Chitta Baral, Peyman Heidari

**Abstract**: Hybrid retrievers can take advantage of both sparse and dense retrievers. Previous hybrid retrievers leverage indexing-heavy dense retrievers. In this work, we study "Is it possible to reduce the indexing memory of hybrid retrievers without sacrificing performance"? Driven by this question, we leverage an indexing-efficient dense retriever (i.e. DrBoost) and introduce a LITE retriever that further reduces the memory of DrBoost. LITE is jointly trained on contrastive learning and knowledge distillation from DrBoost. Then, we integrate BM25, a sparse retriever, with either LITE or DrBoost to form light hybrid retrievers. Our Hybrid-LITE retriever saves 13X memory while maintaining 98.0% performance of the hybrid retriever of BM25 and DPR. In addition, we study the generalization capacity of our light hybrid retrievers on out-of-domain dataset and a set of adversarial attacks datasets. Experiments showcase that light hybrid retrievers achieve better generalization performance than individual sparse and dense retrievers. Nevertheless, our analysis shows that there is a large room to improve the robustness of retrievers, suggesting a new research direction.



## **38. Adversarial Laser Spot: Robust and Covert Physical-World Attack to DNNs**

cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2206.01034v2) [paper-pdf](http://arxiv.org/pdf/2206.01034v2)

**Authors**: Chengyin Hu, Yilong Wang, Kalibinuer Tiliwalidi, Wen Li

**Abstract**: Most existing deep neural networks (DNNs) are easily disturbed by slight noise. However, there are few researches on physical attacks by deploying lighting equipment. The light-based physical attacks has excellent covertness, which brings great security risks to many vision-based applications (such as self-driving). Therefore, we propose a light-based physical attack, called adversarial laser spot (AdvLS), which optimizes the physical parameters of laser spots through genetic algorithm to perform physical attacks. It realizes robust and covert physical attack by using low-cost laser equipment. As far as we know, AdvLS is the first light-based physical attack that perform physical attacks in the daytime. A large number of experiments in the digital and physical environments show that AdvLS has excellent robustness and covertness. In addition, through in-depth analysis of the experimental data, we find that the adversarial perturbations generated by AdvLS have superior adversarial attack migration. The experimental results show that AdvLS impose serious interference to advanced DNNs, we call for the attention of the proposed AdvLS. The code of AdvLS is available at: https://github.com/ChengYinHu/AdvLS



## **39. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

cs.LG

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.09241v2) [paper-pdf](http://arxiv.org/pdf/2305.09241v2)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning.



## **40. Adversarial Neon Beam: A Light-based Physical Attack to DNNs**

cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2204.00853v3) [paper-pdf](http://arxiv.org/pdf/2204.00853v3)

**Authors**: Chengyin Hu, Weiwen Shi, Wen Li

**Abstract**: In the physical world, deep neural networks (DNNs) are impacted by light and shadow, which can have a significant effect on their performance. While stickers have traditionally been used as perturbations in most physical attacks, their perturbations can often be easily detected. To address this, some studies have explored the use of light-based perturbations, such as lasers or projectors, to generate more subtle perturbations, which are artificial rather than natural. In this study, we introduce a novel light-based attack called the adversarial neon beam (AdvNB), which utilizes common neon beams to create a natural black-box physical attack. Our approach is evaluated on three key criteria: effectiveness, stealthiness, and robustness. Quantitative results obtained in simulated environments demonstrate the effectiveness of the proposed method, and in physical scenarios, we achieve an attack success rate of 81.82%, surpassing the baseline. By using common neon beams as perturbations, we enhance the stealthiness of the proposed attack, enabling physical samples to appear more natural. Moreover, we validate the robustness of our approach by successfully attacking advanced DNNs with a success rate of over 75% in all cases. We also discuss defense strategies against the AdvNB attack and put forward other light-based physical attacks.



## **41. Enhancing Accuracy and Robustness through Adversarial Training in Class Incremental Continual Learning**

cs.LG

9 pages, 6 figures

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13678v1) [paper-pdf](http://arxiv.org/pdf/2305.13678v1)

**Authors**: Minchan Kwon, Kangil Kim

**Abstract**: In real life, adversarial attack to deep learning models is a fatal security issue. However, the issue has been rarely discussed in a widely used class-incremental continual learning (CICL). In this paper, we address problems of applying adversarial training to CICL, which is well-known defense method against adversarial attack. A well-known problem of CICL is class-imbalance that biases a model to the current task by a few samples of previous tasks. Meeting with the adversarial training, the imbalance causes another imbalance of attack trials over tasks. Lacking clean data of a minority class by the class-imbalance and increasing of attack trials from a majority class by the secondary imbalance, adversarial training distorts optimal decision boundaries. The distortion eventually decreases both accuracy and robustness than adversarial training. To exclude the effects, we propose a straightforward but significantly effective method, External Adversarial Training (EAT) which can be applied to methods using experience replay. This method conduct adversarial training to an auxiliary external model for the current task data at each time step, and applies generated adversarial examples to train the target model. We verify the effects on a toy problem and show significance on CICL benchmarks of image classification. We expect that the results will be used as the first baseline for robustness research of CICL.



## **42. Adversarial Defenses via Vector Quantization**

cs.LG

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13651v1) [paper-pdf](http://arxiv.org/pdf/2305.13651v1)

**Authors**: Zhiyi Dong, Yongyi Mao

**Abstract**: Building upon Randomized Discretization, we develop two novel adversarial defenses against white-box PGD attacks, utilizing vector quantization in higher dimensional spaces. These methods, termed pRD and swRD, not only offer a theoretical guarantee in terms of certified accuracy, they are also shown, via abundant experiments, to perform comparably or even superior to the current art of adversarial defenses. These methods can be extended to a version that allows further training of the target classifier and demonstrates further improved performance.



## **43. Hardware Trojans in Power Conversion Circuits**

eess.SP

4 pages, 6 figures, will not be submitted to any journals

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13643v1) [paper-pdf](http://arxiv.org/pdf/2305.13643v1)

**Authors**: Jacob Sillman, Ajay Suresh

**Abstract**: This report investigates the potential impact of a Trojan attack on power conversion circuits, specifically a switching signal attack designed to trigger a locking of the pulse width modulation (PWM) signal that goes to a power field-effect transistor (FET). The first simulation shows that this type of attack can cause severe overvoltage, potentially leading to functional failure. The report proposes a solution using a large bypass capacitor to force signal parity, effectively negating the Trojan circuit. The simulation results demonstrate that the proposed solution can effectively thwart the Trojan attack. However, several caveats must be considered, such as the size of the capacitor, possible current leakage, and the possibility that the solution can be circumvented by an adversary with knowledge of the protection strategy. Overall, the findings suggest that proper protection mechanisms, such as the proposed signal-parity solution, must be considered when designing power conversion circuits to mitigate the risk of Trojan attacks.



## **44. Adversarial Ensemble Training by Jointly Learning Label Dependencies and Member Models**

cs.LG

This paper has been accepted by 19th Inter. Conf. on Intelligent  Computing (ICIC 2023)

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2206.14477v3) [paper-pdf](http://arxiv.org/pdf/2206.14477v3)

**Authors**: Lele Wang, Bin Liu

**Abstract**: Training an ensemble of diverse sub-models has been empirically demonstrated as an effective strategy for improving the adversarial robustness of deep neural networks. However, current ensemble training methods for image recognition typically encode image labels using one-hot vectors, which overlook dependency relationships between the labels. In this paper, we propose a novel adversarial en-semble training approach that jointly learns the label dependencies and member models. Our approach adaptively exploits the learned label dependencies to pro-mote diversity among the member models. We evaluate our approach on widely used datasets including MNIST, FashionMNIST, and CIFAR-10, and show that it achieves superior robustness against black-box attacks compared to state-of-the-art methods. Our code is available at https://github.com/ZJLAB-AMMI/LSD.



## **45. Adversarial Infrared Blocks: A Black-box Attack to Thermal Infrared Detectors at Multiple Angles in Physical World**

cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2304.10712v2) [paper-pdf](http://arxiv.org/pdf/2304.10712v2)

**Authors**: Chengyin Hu, Weiwen Shi, Tingsong Jiang, Wen Yao, Ling Tian, Xiaoqian Chen

**Abstract**: Infrared imaging systems have a vast array of potential applications in pedestrian detection and autonomous driving, and their safety performance is of great concern. However, few studies have explored the safety of infrared imaging systems in real-world settings. Previous research has used physical perturbations such as small bulbs and thermal "QR codes" to attack infrared imaging detectors, but such methods are highly visible and lack stealthiness. Other researchers have used hot and cold blocks to deceive infrared imaging detectors, but this method is limited in its ability to execute attacks from various angles. To address these shortcomings, we propose a novel physical attack called adversarial infrared blocks (AdvIB). By optimizing the physical parameters of the adversarial infrared blocks, this method can execute a stealthy black-box attack on thermal imaging system from various angles. We evaluate the proposed method based on its effectiveness, stealthiness, and robustness. Our physical tests show that the proposed method achieves a success rate of over 80% under most distance and angle conditions, validating its effectiveness. For stealthiness, our method involves attaching the adversarial infrared block to the inside of clothing, enhancing its stealthiness. Additionally, we test the proposed method on advanced detectors, and experimental results demonstrate an average attack success rate of 51.2%, proving its robustness. Overall, our proposed AdvIB method offers a promising avenue for conducting stealthy, effective and robust black-box attacks on thermal imaging system, with potential implications for real-world safety and security applications.



## **46. DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection**

cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13625v1) [paper-pdf](http://arxiv.org/pdf/2305.13625v1)

**Authors**: Jiang Liu, Chun Pong Lau, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.



## **47. Attribute-Guided Encryption with Facial Texture Masking**

cs.CV

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.13548v1) [paper-pdf](http://arxiv.org/pdf/2305.13548v1)

**Authors**: Chun Pong Lau, Jiang Liu, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from unauthorized FR systems utilizing adversarial attacks to generate encrypted face images to protect users from being identified by FR systems. However, existing methods suffer from poor visual quality or low attack success rates, which limit their usability in practice. In this paper, we propose Attribute Guided Encryption with Facial Texture Masking (AGE-FTM) that performs a dual manifold adversarial attack on FR systems to achieve both good visual quality and high black box attack success rates. In particular, AGE-FTM utilizes a high fidelity generative adversarial network (GAN) to generate natural on-manifold adversarial samples by modifying facial attributes, and performs the facial texture masking attack to generate imperceptible off-manifold adversarial samples. Extensive experiments on the CelebA-HQ dataset demonstrate that our proposed method produces more natural-looking encrypted images than state-of-the-art methods while achieving competitive attack performance. We further evaluate the effectiveness of AGE-FTM in the real world using a commercial FR API and validate its usefulness in practice through an user study.



## **48. And/or trade-off in artificial neurons: impact on adversarial robustness**

cs.LG

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2102.07389v3) [paper-pdf](http://arxiv.org/pdf/2102.07389v3)

**Authors**: Alessandro Fontana

**Abstract**: Despite the success of neural networks, the issue of classification robustness remains, particularly highlighted by adversarial examples. In this paper, we address this challenge by focusing on the continuum of functions implemented in artificial neurons, ranging from pure AND gates to pure OR gates. Our hypothesis is that the presence of a sufficient number of OR-like neurons in a network can lead to classification brittleness and increased vulnerability to adversarial attacks. We define AND-like neurons and propose measures to increase their proportion in the network. These measures involve rescaling inputs to the [-1,1] interval and reducing the number of points in the steepest section of the sigmoidal activation function. A crucial component of our method is the comparison between a neuron's output distribution when fed with the actual dataset and a randomised version called the "scrambled dataset." Experimental results on the MNIST dataset suggest that our approach holds promise as a direction for further exploration.



## **49. Adversarial Nibbler: A Data-Centric Challenge for Improving the Safety of Text-to-Image Models**

cs.LG

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.14384v1) [paper-pdf](http://arxiv.org/pdf/2305.14384v1)

**Authors**: Alicia Parrish, Hannah Rose Kirk, Jessica Quaye, Charvi Rastogi, Max Bartolo, Oana Inel, Juan Ciro, Rafael Mosquera, Addison Howard, Will Cukierski, D. Sculley, Vijay Janapa Reddi, Lora Aroyo

**Abstract**: The generative AI revolution in recent years has been spurred by an expansion in compute power and data quantity, which together enable extensive pre-training of powerful text-to-image (T2I) models. With their greater capabilities to generate realistic and creative content, these T2I models like DALL-E, MidJourney, Imagen or Stable Diffusion are reaching ever wider audiences. Any unsafe behaviors inherited from pretraining on uncurated internet-scraped datasets thus have the potential to cause wide-reaching harm, for example, through generated images which are violent, sexually explicit, or contain biased and derogatory stereotypes. Despite this risk of harm, we lack systematic and structured evaluation datasets to scrutinize model behavior, especially adversarial attacks that bypass existing safety filters. A typical bottleneck in safety evaluation is achieving a wide coverage of different types of challenging examples in the evaluation set, i.e., identifying 'unknown unknowns' or long-tail problems. To address this need, we introduce the Adversarial Nibbler challenge. The goal of this challenge is to crowdsource a diverse set of failure modes and reward challenge participants for successfully finding safety vulnerabilities in current state-of-the-art T2I models. Ultimately, we aim to provide greater awareness of these issues and assist developers in improving the future safety and reliability of generative AI models. Adversarial Nibbler is a data-centric challenge, part of the DataPerf challenge suite, organized and supported by Kaggle and MLCommons.



## **50. Analyzing the Shuffle Model through the Lens of Quantitative Information Flow**

cs.CR

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.13075v1) [paper-pdf](http://arxiv.org/pdf/2305.13075v1)

**Authors**: Mireya Jurado, Ramon G. Gonze, Mário S. Alvim, Catuscia Palamidessi

**Abstract**: Local differential privacy (LDP) is a variant of differential privacy (DP) that avoids the need for a trusted central curator, at the cost of a worse trade-off between privacy and utility. The shuffle model is a way to provide greater anonymity to users by randomly permuting their messages, so that the link between users and their reported values is lost to the data collector. By combining an LDP mechanism with a shuffler, privacy can be improved at no cost for the accuracy of operations insensitive to permutations, thereby improving utility in many tasks. However, the privacy implications of shuffling are not always immediately evident, and derivations of privacy bounds are made on a case-by-case basis.   In this paper, we analyze the combination of LDP with shuffling in the rigorous framework of quantitative information flow (QIF), and reason about the resulting resilience to inference attacks. QIF naturally captures randomization mechanisms as information-theoretic channels, thus allowing for precise modeling of a variety of inference attacks in a natural way and for measuring the leakage of private information under these attacks. We exploit symmetries of the particular combination of k-RR mechanisms with the shuffle model to achieve closed formulas that express leakage exactly. In particular, we provide formulas that show how shuffling improves protection against leaks in the local model, and study how leakage behaves for various values of the privacy parameter of the LDP mechanism.   In contrast to the strong adversary from differential privacy, we focus on an uninformed adversary, who does not know the value of any individual in the dataset. This adversary is often more realistic as a consumer of statistical datasets, and we show that in some situations mechanisms that are equivalent w.r.t. the strong adversary can provide different privacy guarantees under the uninformed one.



