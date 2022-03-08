# Latest Adversarial Attack Papers
**update at 2022-03-09 06:31:55**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Adversarial Texture for Fooling Person Detectors in the Physical World**

cs.CV

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03373v1)

**Authors**: Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Xiaolin Hu, Fuchun Sun, Bo Zhang

**Abstracts**: Nowadays, cameras equipped with AI systems can capture and analyze images to detect people automatically. However, the AI system can make mistakes when receiving deliberately designed patterns in the real world, i.e., physical adversarial examples. Prior works have shown that it is possible to print adversarial patches on clothes to evade DNN-based person detectors. However, these adversarial examples could have catastrophic drops in the attack success rate when the viewing angle (i.e., the camera's angle towards the object) changes. To perform a multi-angle attack, we propose Adversarial Texture (AdvTexture). AdvTexture can cover clothes with arbitrary shapes so that people wearing such clothes can hide from person detectors from different viewing angles. We propose a generative method, named Toroidal-Cropping-based Expandable Generative Attack (TC-EGA), to craft AdvTexture with repetitive structures. We printed several pieces of cloth with AdvTexure and then made T-shirts, skirts, and dresses in the physical world. Experiments showed that these clothes could fool person detectors in the physical world.



## **2. Uncertify: Attacks Against Neural Network Certification**

cs.LG

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2108.11299v2)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstracts**: Certifiers for neural networks have made great progress towards provable robustness guarantees against evasion attacks using adversarial examples. However, introducing certifiers into deep learning systems also opens up new attack vectors, which need to be considered before deployment. In this work, we conduct the first systematic analysis of training-time attacks against certifiers in practical application pipelines, identifying new threat vectors that can be exploited to degrade the overall system. Using these insights, we design two backdoor attacks against network certifiers, which can drastically reduce certified robustness. For example, adding 1% poisoned data points during training is sufficient to reduce certified robustness by up to 95 percentage points, effectively rendering the certifier useless. We analyze how such novel attacks can compromise the overall system's integrity or availability. Our extensive experiments across multiple datasets, model architectures, and certifiers demonstrate the wide applicability of these attacks. A first investigation into potential defenses shows that current approaches are insufficient to mitigate the issue, highlighting the need for new, more specific solutions.



## **3. The Dangerous Combo: Fileless Malware and Cryptojacking**

cs.CR

9 Pages - Accepted to be published in SoutheastCon 2022 IEEE Region 3  Technical, Professional, and Student Conference. Mobile, Alabama, USA. Mar  31st to Apr 03rd 2022. https://ieeesoutheastcon.org/

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03175v1)

**Authors**: Said Varlioglu, Nelly Elsayed, Zag ElSayed, Murat Ozer

**Abstracts**: Fileless malware and cryptojacking attacks have appeared independently as the new alarming threats in 2017. After 2020, fileless attacks have been devastating for victim organizations with low-observable characteristics. Also, the amount of unauthorized cryptocurrency mining has increased after 2019. Adversaries have started to merge these two different cyberattacks to gain more invisibility and profit under "Fileless Cryptojacking." This paper aims to provide a literature review in academic papers and industry reports for this new threat. Additionally, we present a new threat hunting-oriented DFIR approach with the best practices derived from field experience as well as the literature. Last, this paper reviews the fundamentals of the fileless threat that can also help ransomware researchers examine similar patterns.



## **4. Searching for Robust Neural Architectures via Comprehensive and Reliable Evaluation**

cs.LG

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03128v1)

**Authors**: Jialiang Sun, Tingsong Jiang, Chao Li, Weien Zhou, Xiaoya Zhang, Wen Yao, Xiaoqian Chen

**Abstracts**: Neural architecture search (NAS) could help search for robust network architectures, where defining robustness evaluation metrics is the important procedure. However, current robustness evaluations in NAS are not sufficiently comprehensive and reliable. In particular, the common practice only considers adversarial noise and quantified metrics such as the Jacobian matrix, whereas, some studies indicated that the models are also vulnerable to other types of noises such as natural noise. In addition, existing methods taking adversarial noise as the evaluation just use the robust accuracy of the FGSM or PGD, but these adversarial attacks could not provide the adequately reliable evaluation, leading to the vulnerability of the models under stronger attacks. To alleviate the above problems, we propose a novel framework, called Auto Adversarial Attack and Defense (AAAD), where we employ neural architecture search methods, and four types of robustness evaluations are considered, including adversarial noise, natural noise, system noise and quantified metrics, thereby assisting in finding more robust architectures. Also, among the adversarial noise, we use the composite adversarial attack obtained by random search as the new metric to evaluate the robustness of the model architectures. The empirical results on the CIFAR10 dataset show that the searched efficient attack could help find more robust architectures.



## **5. Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer**

cs.CV

Accepted by CVPR2022, NOT the camera-ready version

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03121v1)

**Authors**: Shengshan Hu, Xiaogeng Liu, Yechao Zhang, Minghui Li, Leo Yu Zhang, Hai Jin, Libing Wu

**Abstracts**: While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality. In this paper, we propose adversarial makeup transfer GAN (AMT-GAN), a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.



## **6. Can You Hear It? Backdoor Attacks via Ultrasonic Triggers**

cs.CR

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2107.14569v3)

**Authors**: Stefanos Koffas, Jing Xu, Mauro Conti, Stjepan Picek

**Abstracts**: This work explores backdoor attacks for automatic speech recognition systems where we inject inaudible triggers. By doing so, we make the backdoor attack challenging to detect for legitimate users, and thus, potentially more dangerous. We conduct experiments on two versions of a speech dataset and three neural networks and explore the performance of our attack concerning the duration, position, and type of the trigger. Our results indicate that less than 1% of poisoned data is sufficient to deploy a backdoor attack and reach a 100% attack success rate. We observed that short, non-continuous triggers result in highly successful attacks. However, since our trigger is inaudible, it can be as long as possible without raising any suspicions making the attack more effective. Finally, we conducted our attack in actual hardware and saw that an adversary could manipulate inference in an Android application by playing the inaudible trigger over the air.



## **7. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

cs.NE

18 pages, 9 figures, 9 tables and 23 References

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2110.01818v5)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the searchability, convergence efficiency and precision of genetic algorithms. In this paper, a novel improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by 15 test functions. The qualitative results show that, compared with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. The quantitative results show that the algorithm performs superiorly in 13 of the 15 tested functions. The Wilcoxon rank-sum test was used for statistical evaluation, showing the significant advantage of the algorithm at $95\%$ confidence intervals. Finally, the algorithm is applied to neural network adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.



## **8. Finding Dynamics Preserving Adversarial Winning Tickets**

cs.LG

Accepted by AISTATS2022

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2202.06488v3)

**Authors**: Xupeng Shi, Pengfei Zheng, A. Adam Ding, Yuan Gao, Weizhong Zhang

**Abstracts**: Modern deep neural networks (DNNs) are vulnerable to adversarial attacks and adversarial training has been shown to be a promising method for improving the adversarial robustness of DNNs. Pruning methods have been considered in adversarial context to reduce model capacity and improve adversarial robustness simultaneously in training. Existing adversarial pruning methods generally mimic the classical pruning methods for natural training, which follow the three-stage 'training-pruning-fine-tuning' pipelines. We observe that such pruning methods do not necessarily preserve the dynamics of dense networks, making it potentially hard to be fine-tuned to compensate the accuracy degradation in pruning. Based on recent works of \textit{Neural Tangent Kernel} (NTK), we systematically study the dynamics of adversarial training and prove the existence of trainable sparse sub-network at initialization which can be trained to be adversarial robust from scratch. This theoretically verifies the \textit{lottery ticket hypothesis} in adversarial context and we refer such sub-network structure as \textit{Adversarial Winning Ticket} (AWT). We also show empirical evidences that AWT preserves the dynamics of adversarial training and achieve equal performance as dense adversarial training.



## **9. aaeCAPTCHA: The Design and Implementation of Audio Adversarial CAPTCHA**

cs.CR

Accepted at 7th IEEE European Symposium on Security and Privacy  (EuroS&P 2022)

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.02735v1)

**Authors**: Md Imran Hossen, Xiali Hei

**Abstracts**: CAPTCHAs are designed to prevent malicious bot programs from abusing websites. Most online service providers deploy audio CAPTCHAs as an alternative to text and image CAPTCHAs for visually impaired users. However, prior research investigating the security of audio CAPTCHAs found them highly vulnerable to automated attacks using Automatic Speech Recognition (ASR) systems. To improve the robustness of audio CAPTCHAs against automated abuses, we present the design and implementation of an audio adversarial CAPTCHA (aaeCAPTCHA) system in this paper. The aaeCAPTCHA system exploits audio adversarial examples as CAPTCHAs to prevent the ASR systems from automatically solving them. Furthermore, we conducted a rigorous security evaluation of our new audio CAPTCHA design against five state-of-the-art DNN-based ASR systems and three commercial Speech-to-Text (STT) services. Our experimental evaluations demonstrate that aaeCAPTCHA is highly secure against these speech recognition technologies, even when the attacker has complete knowledge of the current attacks against audio adversarial examples. We also conducted a usability evaluation of the proof-of-concept implementation of the aaeCAPTCHA scheme. Our results show that it achieves high robustness at a moderate usability cost compared to normal audio CAPTCHAs. Finally, our extensive analysis highlights that aaeCAPTCHA can significantly enhance the security and robustness of traditional audio CAPTCHA systems while maintaining similar usability.



## **10. Generating Out of Distribution Adversarial Attack using Latent Space Poisoning**

cs.CV

IEEE SPL 2021

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2012.05027v2)

**Authors**: Ujjwal Upadhyay, Prerana Mukherjee

**Abstracts**: Traditional adversarial attacks rely upon the perturbations generated by gradients from the network which are generally safeguarded by gradient guided search to provide an adversarial counterpart to the network. In this paper, we propose a novel mechanism of generating adversarial examples where the actual image is not corrupted rather its latent space representation is utilized to tamper with the inherent structure of the image while maintaining the perceptual quality intact and to act as legitimate data samples. As opposed to gradient-based attacks, the latent space poisoning exploits the inclination of classifiers to model the independent and identical distribution of the training dataset and tricks it by producing out of distribution samples. We train a disentangled variational autoencoder (beta-VAE) to model the data in latent space and then we add noise perturbations using a class-conditioned distribution function to the latent space under the constraint that it is misclassified to the target label. Our empirical results on MNIST, SVHN, and CelebA dataset validate that the generated adversarial examples can easily fool robust l_0, l_2, l_inf norm classifiers designed using provably robust defense mechanisms.



## **11. Adversarial samples for deep monocular 6D object pose estimation**

cs.CV

15 pages

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.00302v2)

**Authors**: Jinlai Zhang, Weiming Li, Shuang Liang, Hao Wang, Jihong Zhu

**Abstracts**: Estimating 6D object pose from an RGB image is important for many real-world applications such as autonomous driving and robotic grasping. Recent deep learning models have achieved significant progress on this task but their robustness received little research attention. In this work, for the first time, we study adversarial samples that can fool deep learning models with imperceptible perturbations to input image. In particular, we propose a Unified 6D pose estimation Attack, namely U6DA, which can successfully attack several state-of-the-art (SOTA) deep learning models for 6D pose estimation. The key idea of our U6DA is to fool the models to predict wrong results for object instance localization and shape that are essential for correct 6D pose estimation. Specifically, we explore a transfer-based black-box attack to 6D pose estimation. We design the U6DA loss to guide the generation of adversarial examples, the loss aims to shift the segmentation attention map away from its original position. We show that the generated adversarial samples are not only effective for direct 6D pose estimation models, but also are able to attack two-stage models regardless of their robust RANSAC modules. Extensive experiments were conducted to demonstrate the effectiveness, transferability, and anti-defense capability of our U6DA on large-scale public benchmarks. We also introduce a new U6DA-Linemod dataset for robustness study of the 6D pose estimation task. Our codes and dataset will be available at \url{https://github.com/cuge1995/U6DA}.



## **12. Training privacy-preserving video analytics pipelines by suppressing features that reveal information about private attributes**

cs.CV

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.02635v1)

**Authors**: Chau Yi Li, Andrea Cavallaro

**Abstracts**: Deep neural networks are increasingly deployed for scene analytics, including to evaluate the attention and reaction of people exposed to out-of-home advertisements. However, the features extracted by a deep neural network that was trained to predict a specific, consensual attribute (e.g. emotion) may also encode and thus reveal information about private, protected attributes (e.g. age or gender). In this work, we focus on such leakage of private information at inference time. We consider an adversary with access to the features extracted by the layers of a deployed neural network and use these features to predict private attributes. To prevent the success of such an attack, we modify the training of the network using a confusion loss that encourages the extraction of features that make it difficult for the adversary to accurately predict private attributes. We validate this training approach on image-based tasks using a publicly available dataset. Results show that, compared to the original network, the proposed PrivateNet can reduce the leakage of private information of a state-of-the-art emotion recognition classifier by 2.88% for gender and by 13.06% for age group, with a minimal effect on task accuracy.



## **13. Optimal Clock Synchronization with Signatures**

cs.DC

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02553v1)

**Authors**: Christoph Lenzen, Julian Loss

**Abstracts**: Cryptographic signatures can be used to increase the resilience of distributed systems against adversarial attacks, by increasing the number of faulty parties that can be tolerated. While this is well-studied for consensus, it has been underexplored in the context of fault-tolerant clock synchronization, even in fully connected systems. Here, the honest parties of an $n$-node system are required to compute output clocks of small skew (i.e., maximum phase offset) despite local clock rates varying between $1$ and $\vartheta>1$, end-to-end communication delays varying between $d-u$ and $d$, and the interference from malicious parties. So far, it is only known that clock pulses of skew $d$ can be generated with (trivially optimal) resilience of $\lceil n/2\rceil-1$ (PODC `19), improving over the tight bound of $\lceil n/3\rceil-1$ holding without signatures for \emph{any} skew bound (STOC `84, PODC `85). Since typically $d\gg u$ and $\vartheta-1\ll 1$, this is far from the lower bound of $u+(\vartheta-1)d$ that applies even in the fault-free case (IPL `01).   We prove matching upper and lower bounds of $\Theta(u+(\vartheta-1)d)$ on the skew for the resilience range from $\lceil n/3\rceil$ to $\lceil n/2\rceil-1$. The algorithm showing the upper bound is, under the assumption that the adversary cannot forge signatures, deterministic. The lower bound holds even if clocks are initially perfectly synchronized, message delays between honest nodes are known, $\vartheta$ is arbitrarily close to one, and the synchronization algorithm is randomized. This has crucial implications for network designers that seek to leverage signatures for providing more robust time. In contrast to the setting without signatures, they must ensure that an attacker cannot easily bypass the lower bound on the delay on links with a faulty endpoint.



## **14. Medical Aegis: Robust adversarial protectors for medical images**

cs.CV

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2111.10969v4)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.



## **15. Adversarial Patterns: Building Robust Android Malware Classifiers**

cs.CR

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02121v1)

**Authors**: Dipkamal Bhusal, Nidhi Rastogi

**Abstracts**: Deep learning-based classifiers have substantially improved recognition of malware samples. However, these classifiers can be vulnerable to adversarial input perturbations. Any vulnerability in malware classifiers poses significant threats to the platforms they defend. Therefore, to create stronger defense models against malware, we must understand the patterns in input perturbations caused by an adversary. This survey paper presents a comprehensive study on adversarial machine learning for android malware classifiers. We first present an extensive background in building a machine learning classifier for android malware, covering both image-based and text-based feature extraction approaches. Then, we examine the pattern and advancements in the state-of-the-art research in evasion attacks and defenses. Finally, we present guidelines for designing robust malware classifiers and enlist research directions for the future.



## **16. Label Leakage and Protection from Forward Embedding in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.01451v2)

**Authors**: Jiankai Sun, Xin Yang, Yuanshun Yao, Chong Wang

**Abstracts**: Vertical federated learning (vFL) has gained much attention and been deployed to solve machine learning problems with data privacy concerns in recent years. However, some recent work demonstrated that vFL is vulnerable to privacy leakage even though only the forward intermediate embedding (rather than raw features) and backpropagated gradients (rather than raw labels) are communicated between the involved participants. As the raw labels often contain highly sensitive information, some recent work has been proposed to prevent the label leakage from the backpropagated gradients effectively in vFL. However, these work only identified and defended the threat of label leakage from the backpropagated gradients. None of these work has paid attention to the problem of label leakage from the intermediate embedding. In this paper, we propose a practical label inference method which can steal private labels effectively from the shared intermediate embedding even though some existing protection methods such as label differential privacy and gradients perturbation are applied. The effectiveness of the label attack is inseparable from the correlation between the intermediate embedding and corresponding private labels. To mitigate the issue of label leakage from the forward embedding, we add an additional optimization goal at the label party to limit the label stealing ability of the adversary by minimizing the distance correlation between the intermediate embedding and corresponding private labels. We conducted massive experiments to demonstrate the effectiveness of our proposed protection methods.



## **17. Differentially Private Label Protection in Split Learning**

cs.LG

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02073v1)

**Authors**: Xin Yang, Jiankai Sun, Yuanshun Yao, Junyuan Xie, Chong Wang

**Abstracts**: Split learning is a distributed training framework that allows multiple parties to jointly train a machine learning model over vertically partitioned data (partitioned by attributes). The idea is that only intermediate computation results, rather than private features and labels, are shared between parties so that raw training data remains private. Nevertheless, recent works showed that the plaintext implementation of split learning suffers from severe privacy risks that a semi-honest adversary can easily reconstruct labels. In this work, we propose \textsf{TPSL} (Transcript Private Split Learning), a generic gradient perturbation based split learning framework that provides provable differential privacy guarantee. Differential privacy is enforced on not only the model weights, but also the communicated messages in the distributed computation setting. Our experiments on large-scale real-world datasets demonstrate the robustness and effectiveness of \textsf{TPSL} against label leakage attacks. We also find that \textsf{TPSL} have a better utility-privacy trade-off than baselines.



## **18. Can Authoritative Governments Abuse the Right to Access?**

cs.CR

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02068v1)

**Authors**: Cédric Lauradoux

**Abstracts**: The right to access is a great tool provided by the GDPR to empower data subjects with their data. However, it needs to be implemented properly otherwise it could turn subject access requests against the subjects privacy. Indeed, recent works have shown that it is possible to abuse the right to access using impersonation attacks. We propose to extend those impersonation attacks by considering that the adversary has an access to governmental resources. In this case, the adversary can forge official documents or exploit copy of them. Our attack affects more people than one may expect. To defeat the attacks from this kind of adversary, several solutions are available like multi-factors or proof of aliveness. Our attacks highlight the need for strong procedures to authenticate subject access requests.



## **19. Autonomous and Resilient Control for Optimal LEO Satellite Constellation Coverage Against Space Threats**

eess.SY

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02050v1)

**Authors**: Yuhan Zhao, Quanyan Zhu

**Abstracts**: LEO satellite constellation coverage has served as the base platform for various space applications. However, the rapidly evolving security environment such as orbit debris and adversarial space threats are greatly endangering the security of satellite constellation and integrity of the satellite constellation coverage. As on-orbit repairs are challenging, a distributed and autonomous protection mechanism is necessary to ensure the adaptation and self-healing of the satellite constellation coverage from different attacks. To this end, we establish an integrative and distributed framework to enable resilient satellite constellation coverage planning and control in a single orbit. Each satellite can make decisions individually to recover from adversarial and non-adversarial attacks and keep providing coverage service. We first provide models and methodologies to measure the coverage performance. Then, we formulate the joint resilient coverage planning-control problem as a two-stage problem. A coverage game is proposed to find the equilibrium constellation deployment for resilient coverage planning and an agent-based algorithm is developed to compute the equilibrium. The multi-waypoint Model Predictive Control (MPC) methodology is adopted to achieve autonomous self-healing control. Finally, we use a typical LEO satellite constellation as a case study to corroborate the results.



## **20. Why adversarial training can hurt robust accuracy**

cs.LG

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02006v1)

**Authors**: Jacob Clarysse, Julia Hörmann, Fanny Yang

**Abstracts**: Machine learning classifiers with high test accuracy often perform poorly under adversarial attacks. It is commonly believed that adversarial training alleviates this issue. In this paper, we demonstrate that, surprisingly, the opposite may be true -- Even though adversarial training helps when enough data is available, it may hurt robust generalization in the small sample size regime. We first prove this phenomenon for a high-dimensional linear classification setting with noiseless observations. Our proof provides explanatory insights that may also transfer to feature learning models. Further, we observe in experiments on standard image datasets that the same behavior occurs for perceptible attacks that effectively reduce class information such as mask attacks and object corruptions.



## **21. Dynamic Backdoor Attacks Against Machine Learning Models**

cs.CR

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2003.03675v2)

**Authors**: Ahmed Salem, Rui Wen, Michael Backes, Shiqing Ma, Yang Zhang

**Abstracts**: Machine learning (ML) has made tremendous progress during the past decade and is being adopted in various critical real-world applications. However, recent research has shown that ML models are vulnerable to multiple security and privacy attacks. In particular, backdoor attacks against ML models have recently raised a lot of awareness. A successful backdoor attack can cause severe consequences, such as allowing an adversary to bypass critical authentication systems.   Current backdooring techniques rely on adding static triggers (with fixed patterns and locations) on ML model inputs which are prone to detection by the current backdoor detection mechanisms. In this paper, we propose the first class of dynamic backdooring techniques against deep neural networks (DNN), namely Random Backdoor, Backdoor Generating Network (BaN), and conditional Backdoor Generating Network (c-BaN). Triggers generated by our techniques can have random patterns and locations, which reduce the efficacy of the current backdoor detection mechanisms. In particular, BaN and c-BaN based on a novel generative network are the first two schemes that algorithmically generate triggers. Moreover, c-BaN is the first conditional backdooring technique that given a target label, it can generate a target-specific trigger. Both BaN and c-BaN are essentially a general framework which renders the adversary the flexibility for further customizing backdoor attacks.   We extensively evaluate our techniques on three benchmark datasets: MNIST, CelebA, and CIFAR-10. Our techniques achieve almost perfect attack performance on backdoored data with a negligible utility loss. We further show that our techniques can bypass current state-of-the-art defense mechanisms against backdoor attacks, including ABS, Februus, MNTD, Neural Cleanse, and STRIP.



## **22. Assessing the Robustness of Visual Question Answering Models**

cs.CV

24 pages, 13 figures, International Journal of Computer Vision (IJCV)  [under review]. arXiv admin note: substantial text overlap with  arXiv:1711.06232, arXiv:1709.04625

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/1912.01452v2)

**Authors**: Jia-Hong Huang, Modar Alfadly, Bernard Ghanem, Marcel Worring

**Abstracts**: Deep neural networks have been playing an essential role in the task of Visual Question Answering (VQA). Until recently, their accuracy has been the main focus of research. Now there is a trend toward assessing the robustness of these models against adversarial attacks by evaluating the accuracy of these models under increasing levels of noisiness in the inputs of VQA models. In VQA, the attack can target the image and/or the proposed query question, dubbed main question, and yet there is a lack of proper analysis of this aspect of VQA. In this work, we propose a new method that uses semantically related questions, dubbed basic questions, acting as noise to evaluate the robustness of VQA models. We hypothesize that as the similarity of a basic question to the main question decreases, the level of noise increases. To generate a reasonable noise level for a given main question, we rank a pool of basic questions based on their similarity with this main question. We cast this ranking problem as a LASSO optimization problem. We also propose a novel robustness measure Rscore and two large-scale basic question datasets in order to standardize robustness analysis of VQA models. The experimental results demonstrate that the proposed evaluation method is able to effectively analyze the robustness of VQA models. To foster the VQA research, we will publish our proposed datasets.



## **23. Detection of Word Adversarial Examples in Text Classification: Benchmark and Baseline via Robust Density Estimation**

cs.CL

Findings of ACL 2022

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.01677v1)

**Authors**: KiYoon Yoo, Jangho Kim, Jiho Jang, Nojun Kwak

**Abstracts**: Word-level adversarial attacks have shown success in NLP models, drastically decreasing the performance of transformer-based models in recent years. As a countermeasure, adversarial defense has been explored, but relatively few efforts have been made to detect adversarial examples. However, detecting adversarial examples may be crucial for automated tasks (e.g. review sentiment analysis) that wish to amass information about a certain population and additionally be a step towards a robust defense system. To this end, we release a dataset for four popular attack methods on four datasets and four models to encourage further research in this field. Along with it, we propose a competitive baseline based on density estimation that has the highest AUC on 29 out of 30 dataset-attack-model combinations. Source code is available in https://github.com/anoymous92874838/text-adv-detection.



## **24. On Improving Adversarial Transferability of Vision Transformers**

cs.CV

ICLR'22 (Spotlight), the first two authors contributed equally. Code:  https://t.ly/hBbW

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2106.04169v3)

**Authors**: Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Fahad Shahbaz Khan, Fatih Porikli

**Abstracts**: Vision transformers (ViTs) process input images as sequences of patches via self-attention; a radically different architecture than convolutional neural networks (CNNs). This makes it interesting to study the adversarial feature space of ViT models and their transferability. In particular, we observe that adversarial patterns found via conventional adversarial attacks show very \emph{low} black-box transferability even for large ViT models. We show that this phenomenon is only due to the sub-optimal attack procedures that do not leverage the true representation potential of ViTs. A deep ViT is composed of multiple blocks, with a consistent architecture comprising of self-attention and feed-forward layers, where each block is capable of independently producing a class token. Formulating an attack using only the last class token (conventional approach) does not directly leverage the discriminative information stored in the earlier tokens, leading to poor adversarial transferability of ViTs. Using the compositional nature of ViT models, we enhance transferability of existing attacks by introducing two novel strategies specific to the architecture of ViT models. (i) Self-Ensemble: We propose a method to find multiple discriminative pathways by dissecting a single ViT model into an ensemble of networks. This allows explicitly utilizing class-specific information at each ViT block. (ii) Token Refinement: We then propose to refine the tokens to further enhance the discriminative capacity at each block of ViT. Our token refinement systematically combines the class tokens with structural information preserved within the patch tokens.



## **25. On Robustness of Neural Ordinary Differential Equations**

cs.LG

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/1910.05513v4)

**Authors**: Hanshu Yan, Jiawei Du, Vincent Y. F. Tan, Jiashi Feng

**Abstracts**: Neural ordinary differential equations (ODEs) have been attracting increasing attention in various research domains recently. There have been some works studying optimization issues and approximation capabilities of neural ODEs, but their robustness is still yet unclear. In this work, we fill this important gap by exploring robustness properties of neural ODEs both empirically and theoretically. We first present an empirical study on the robustness of the neural ODE-based networks (ODENets) by exposing them to inputs with various types of perturbations and subsequently investigating the changes of the corresponding outputs. In contrast to conventional convolutional neural networks (CNNs), we find that the ODENets are more robust against both random Gaussian perturbations and adversarial attack examples. We then provide an insightful understanding of this phenomenon by exploiting a certain desirable property of the flow of a continuous-time ODE, namely that integral curves are non-intersecting. Our work suggests that, due to their intrinsic robustness, it is promising to use neural ODEs as a basic block for building robust deep network models. To further enhance the robustness of vanilla neural ODEs, we propose the time-invariant steady neural ODE (TisODE), which regularizes the flow on perturbed data via the time-invariant property and the imposition of a steady-state constraint. We show that the TisODE method outperforms vanilla neural ODEs and also can work in conjunction with other state-of-the-art architectural methods to build more robust deep networks.



## **26. Authentication Attacks on Projection-based Cancelable Biometric Schemes**

cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2110.15163v2)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.



## **27. Ad2Attack: Adaptive Adversarial Attack on Real-Time UAV Tracking**

cs.CV

7 pages, 7 figures, accepted by ICRA 2022

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.01516v1)

**Authors**: Changhong Fu, Sihang Li, Xinnan Yuan, Junjie Ye, Ziang Cao, Fangqiang Ding

**Abstracts**: Visual tracking is adopted to extensive unmanned aerial vehicle (UAV)-related applications, which leads to a highly demanding requirement on the robustness of UAV trackers. However, adding imperceptible perturbations can easily fool the tracker and cause tracking failures. This risk is often overlooked and rarely researched at present. Therefore, to help increase awareness of the potential risk and the robustness of UAV tracking, this work proposes a novel adaptive adversarial attack approach, i.e., Ad$^2$Attack, against UAV object tracking. Specifically, adversarial examples are generated online during the resampling of the search patch image, which leads trackers to lose the target in the following frames. Ad$^2$Attack is composed of a direct downsampling module and a super-resolution upsampling module with adaptive stages. A novel optimization function is proposed for balancing the imperceptibility and efficiency of the attack. Comprehensive experiments on several well-known benchmarks and real-world conditions show the effectiveness of our attack method, which dramatically reduces the performance of the most advanced Siamese trackers.



## **28. Two Attacks On Proof-of-Stake GHOST/Ethereum**

cs.CR

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01315v1)

**Authors**: Joachim Neu, Ertem Nusret Tas, David Tse

**Abstracts**: We present two attacks targeting the Proof-of-Stake (PoS) Ethereum consensus protocol. The first attack suggests a fundamental conceptual incompatibility between PoS and the Greedy Heaviest-Observed Sub-Tree (GHOST) fork choice paradigm employed by PoS Ethereum. In a nutshell, PoS allows an adversary with a vanishing amount of stake to produce an unlimited number of equivocating blocks. While most equivocating blocks will be orphaned, such orphaned `uncle blocks' still influence fork choice under the GHOST paradigm, bestowing upon the adversary devastating control over the canonical chain. While the Latest Message Driven (LMD) aspect of current PoS Ethereum prevents a straightforward application of this attack, our second attack shows how LMD specifically can be exploited to obtain a new variant of the balancing attack that overcomes a recent protocol addition that was intended to mitigate balancing-type attacks. Thus, in its current form, PoS Ethereum without and with LMD is vulnerable to our first and second attack, respectively.



## **29. Detecting Adversarial Perturbations in Multi-Task Perception**

cs.CV

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01177v1)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code will be available on github. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.



## **30. How to Inject Backdoors with Better Consistency: Logit Anchoring on Clean Data**

cs.LG

Accepted by ICLR 2022

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2109.01300v2)

**Authors**: Zhiyuan Zhang, Lingjuan Lyu, Weiqiang Wang, Lichao Sun, Xu Sun

**Abstracts**: Since training a large-scale backdoored model from scratch requires a large training dataset, several recent attacks have considered to inject backdoors into a trained clean model without altering model behaviors on the clean data. Previous work finds that backdoors can be injected into a trained clean model with Adversarial Weight Perturbation (AWP). Here AWPs refers to the variations of parameters that are small in backdoor learning. In this work, we observe an interesting phenomenon that the variations of parameters are always AWPs when tuning the trained clean model to inject backdoors. We further provide theoretical analysis to explain this phenomenon. We formulate the behavior of maintaining accuracy on clean data as the consistency of backdoored models, which includes both global consistency and instance-wise consistency. We extensively analyze the effects of AWPs on the consistency of backdoored models. In order to achieve better consistency, we propose a novel anchoring loss to anchor or freeze the model behaviors on the clean data, with a theoretical guarantee. Both the analytical and the empirical results validate the effectiveness of the anchoring loss in improving the consistency, especially the instance-wise consistency.



## **31. Video is All You Need: Attacking PPG-based Biometric Authentication**

cs.CR

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00928v1)

**Authors**: Lin Li, Chao Chen, Lei Pan, Jun Zhang, Yang Xiang

**Abstracts**: Unobservable physiological signals enhance biometric authentication systems. Photoplethysmography (PPG) signals are convenient owning to its ease of measurement and are usually well protected against remote adversaries in authentication. Any leaked PPG signals help adversaries compromise the biometric authentication systems, and the advent of remote PPG (rPPG) enables adversaries to acquire PPG signals through restoration. While potentially dangerous, rPPG-based attacks are overlooked because existing methods require the victim's PPG signals. This paper proposes a novel spoofing attack approach that uses the waveforms of rPPG signals extracted from video clips to fool the PPG-based biometric authentication. We develop a new PPG restoration model that does not require leaked PPG signals for adversarial attacks. Test results on state-of-art PPG-based biometric authentication show that the signals recovered through rPPG pose a severe threat to PPG-based biometric authentication.



## **32. Canonical foliations of neural networks: application to robustness**

stat.ML

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00922v1)

**Authors**: Eliot Tron, Nicolas Couellan, Stéphane Puechmorel

**Abstracts**: Adversarial attack is an emerging threat to the trustability of machine learning. Understanding these attacks is becoming a crucial task. We propose a new vision on neural network robustness using Riemannian geometry and foliation theory, and create a new adversarial attack by taking into account the curvature of the data space. This new adversarial attack called the "dog-leg attack" is a two-step approximation of a geodesic in the data space. The data space is treated as a (pseudo) Riemannian manifold equipped with the pullback of the Fisher Information Metric (FIM) of the neural network. In most cases, this metric is only semi-definite and its kernel becomes a central object to study. A canonical foliation is derived from this kernel. The curvature of the foliation's leaves gives the appropriate correction to get a two-step approximation of the geodesic and hence a new efficient adversarial attack. Our attack is tested on a toy example, a neural network trained to mimic the $\texttt{Xor}$ function, and demonstrates better results that the state of the art attack presented by Zhao et al. (2019).



## **33. MIAShield: Defending Membership Inference Attacks via Preemptive Exclusion of Members**

cs.CR

21 pages, 17 figures, 10 tables

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00915v1)

**Authors**: Ismat Jarin, Birhanu Eshete

**Abstracts**: In membership inference attacks (MIAs), an adversary observes the predictions of a model to determine whether a sample is part of the model's training data. Existing MIA defenses conceal the presence of a target sample through strong regularization, knowledge distillation, confidence masking, or differential privacy.   We propose MIAShield, a new MIA defense based on preemptive exclusion of member samples instead of masking the presence of a member. The key insight in MIAShield is weakening the strong membership signal that stems from the presence of a target sample by preemptively excluding it at prediction time without compromising model utility. To that end, we design and evaluate a suite of preemptive exclusion oracles leveraging model-confidence, exact or approximate sample signature, and learning-based exclusion of member data points. To be practical, MIAShield splits a training data into disjoint subsets and trains each subset to build an ensemble of models. The disjointedness of subsets ensures that a target sample belongs to only one subset, which isolates the sample to facilitate the preemptive exclusion goal.   We evaluate MIAShield on three benchmark image classification datasets. We show that MIAShield effectively mitigates membership inference (near random guess) for a wide range of MIAs, achieves far better privacy-utility trade-off compared with state-of-the-art defenses, and remains resilient against an adaptive adversary.



## **34. ECG-ATK-GAN: Robustness against Adversarial Attacks on ECGs using Conditional Generative Adversarial Networks**

eess.SP

10 pages, 3 figures, 3 tables

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2110.09983v2)

**Authors**: Khondker Fariha Hossain, Sharif Amit Kamran, Alireza Tavakkoli, Xingjun Ma

**Abstracts**: Automating arrhythmia detection from ECG requires a robust and trusted system that retains high accuracy under electrical disturbances. Many machine learning approaches have reached human-level performance in classifying arrhythmia from ECGs. However, these architectures are vulnerable to adversarial attacks, which can misclassify ECG signals by decreasing the model's accuracy. Adversarial attacks are small crafted perturbations injected in the original data which manifest the out-of-distribution shifts in signal to misclassify the correct class. Thus, security concerns arise for false hospitalization and insurance fraud abusing these perturbations. To mitigate this problem, we introduce a novel Conditional Generative Adversarial Network (GAN), robust against adversarial attacked ECG signals and retaining high accuracy. Our architecture integrates a new class-weighted objective function for adversarial perturbation identification and two novel blocks for discerning and combining out-of-distribution shifts in signals in the learning process for accurately classifying various arrhythmia types. Furthermore, we benchmark our architecture on six different white and black-box attacks and compare with other recently proposed arrhythmia classification models on two publicly available ECG arrhythmia datasets. The experiment confirms that our model is more robust against such adversarial attacks for classifying arrhythmia with high accuracy.



## **35. Proceedings of the Artificial Intelligence for Cyber Security (AICS) Workshop at AAAI 2022**

cs.CR

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.14010v2)

**Authors**: James Holt, Edward Raff, Ahmad Ridley, Dennis Ross, Arunesh Sinha, Diane Staheli, William Streilen, Milind Tambe, Yevgeniy Vorobeychik, Allan Wollaber

**Abstracts**: The workshop will focus on the application of AI to problems in cyber security. Cyber systems generate large volumes of data, utilizing this effectively is beyond human capabilities. Additionally, adversaries continue to develop new attacks. Hence, AI methods are required to understand and protect the cyber domain. These challenges are widely studied in enterprise networks, but there are many gaps in research and practice as well as novel problems in other domains.   In general, AI techniques are still not widely adopted in the real world. Reasons include: (1) a lack of certification of AI for security, (2) a lack of formal study of the implications of practical constraints (e.g., power, memory, storage) for AI systems in the cyber domain, (3) known vulnerabilities such as evasion, poisoning attacks, (4) lack of meaningful explanations for security analysts, and (5) lack of analyst trust in AI solutions. There is a need for the research community to develop novel solutions for these practical issues.



## **36. Beyond Gradients: Exploiting Adversarial Priors in Model Inversion Attacks**

cs.LG

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2203.00481v1)

**Authors**: Dmitrii Usynin, Daniel Rueckert, Georgios Kaissis

**Abstracts**: Collaborative machine learning settings like federated learning can be susceptible to adversarial interference and attacks. One class of such attacks is termed model inversion attacks, characterised by the adversary reverse-engineering the model to extract representations and thus disclose the training data. Prior implementations of this attack typically only rely on the captured data (i.e. the shared gradients) and do not exploit the data the adversary themselves control as part of the training consortium. In this work, we propose a novel model inversion framework that builds on the foundations of gradient-based model inversion attacks, but additionally relies on matching the features and the style of the reconstructed image to data that is controlled by an adversary. Our technique outperforms existing gradient-based approaches both qualitatively and quantitatively, while still maintaining the same honest-but-curious threat model, allowing the adversary to obtain enhanced reconstructions while remaining concealed.



## **37. RAB: Provable Robustness Against Backdoor Attacks**

cs.LG

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2003.08904v6)

**Authors**: Maurice Weber, Xiaojun Xu, Bojan Karlaš, Ce Zhang, Bo Li

**Abstracts**: Recent studies have shown that deep neural networks are vulnerable to adversarial attacks, including evasion and backdoor (poisoning) attacks. On the defense side, there have been intensive efforts on improving both empirical and provable robustness against evasion attacks; however, provable robustness against backdoor attacks still remains largely unexplored. In this paper, we focus on certifying the machine learning model robustness against general threat models, especially backdoor attacks. We first provide a unified framework via randomized smoothing techniques and show how it can be instantiated to certify the robustness against both evasion and backdoor attacks. We then propose the first robust training process, RAB, to smooth the trained model and certify its robustness against backdoor attacks. We theoretically prove the robustness bound for machine learning models trained with RAB, and prove that our robustness bound is tight. We derive the robustness conditions for different smoothing distributions including Gaussian and uniform distributions. In addition, we theoretically show that it is possible to train the robust smoothed models efficiently for simple models such as K-nearest neighbor classifiers, and we propose an exact smooth-training algorithm which eliminates the need to sample from a noise distribution for such models. Empirically, we conduct comprehensive experiments for different machine learning models such as DNNs and K-NN models on MNIST, CIFAR-10, and ImageNette datasets and provide the first benchmark for certified robustness against backdoor attacks. In addition, we evaluate K-NN models on a spambase tabular dataset to demonstrate the advantages of the proposed exact algorithm. Both the theoretic analysis and the comprehensive evaluation on diverse ML models and datasets shed lights on further robust learning strategies against general training time attacks.



## **38. Towards Robust Stacked Capsule Autoencoder with Hybrid Adversarial Training**

cs.CV

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.13755v2)

**Authors**: Jiazhu Dai, Siwei Xiong

**Abstracts**: Capsule networks (CapsNets) are new neural networks that classify images based on the spatial relationships of features. By analyzing the pose of features and their relative positions, it is more capable to recognize images after affine transformation. The stacked capsule autoencoder (SCAE) is a state-of-the-art CapsNet, and achieved unsupervised classification of CapsNets for the first time. However, the security vulnerabilities and the robustness of the SCAE has rarely been explored. In this paper, we propose an evasion attack against SCAE, where the attacker can generate adversarial perturbations based on reducing the contribution of the object capsules in SCAE related to the original category of the image. The adversarial perturbations are then applied to the original images, and the perturbed images will be misclassified. Furthermore, we propose a defense method called Hybrid Adversarial Training (HAT) against such evasion attacks. HAT makes use of adversarial training and adversarial distillation to achieve better robustness and stability. We evaluate the defense method and the experimental results show that the refined SCAE model can achieve 82.14% classification accuracy under evasion attack. The source code is available at https://github.com/FrostbiteXSW/SCAE_Defense.



## **39. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

cs.CR

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.12154v2)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a simple trigger and targeting only one class to using many sophisticated triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make out-of-date assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. In this paper, we advocate general defenses that are effective and robust against various Trojan attacks and propose two novel "filtering" defenses with these characteristics called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF). VIF and AIF leverage variational inference and adversarial training respectively to purify all potential Trojan triggers in the input at run time without making any assumption about their numbers and forms. We further extend "filtering" to "filtering-then-contrasting" - a new defense mechanism that helps avoid the drop in classification accuracy on clean data caused by filtering. Extensive experimental results show that our proposed defenses significantly outperform 4 well-known defenses in mitigating 5 different Trojan attacks including the two state-of-the-art which defeat many strong defenses.



## **40. Adversarial Attack Framework on Graph Embedding Models with Limited Knowledge**

cs.LG

Journal extension of GF-Attack, accepted by TKDE

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2105.12419v2)

**Authors**: Heng Chang, Yu Rong, Tingyang Xu, Wenbing Huang, Honglei Zhang, Peng Cui, Xin Wang, Wenwu Zhu, Junzhou Huang

**Abstracts**: With the success of the graph embedding model in both academic and industry areas, the robustness of graph embedding against adversarial attack inevitably becomes a crucial problem in graph learning. Existing works usually perform the attack in a white-box fashion: they need to access the predictions/labels to construct their adversarial loss. However, the inaccessibility of predictions/labels makes the white-box attack impractical to a real graph learning system. This paper promotes current frameworks in a more general and flexible sense -- we demand to attack various kinds of graph embedding models with black-box driven. We investigate the theoretical connections between graph signal processing and graph embedding models and formulate the graph embedding model as a general graph signal process with a corresponding graph filter. Therefore, we design a generalized adversarial attacker: GF-Attack. Without accessing any labels and model predictions, GF-Attack can perform the attack directly on the graph filter in a black-box fashion. We further prove that GF-Attack can perform an effective attack without knowing the number of layers of graph embedding models. To validate the generalization of GF-Attack, we construct the attacker on four popular graph embedding models. Extensive experiments validate the effectiveness of GF-Attack on several benchmark datasets.



## **41. Load-Altering Attacks Against Power Grids under COVID-19 Low-Inertia Conditions**

cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2201.10505v2)

**Authors**: Subhash Lakshminarayana, Juan Ospina, Charalambos Konstantinou

**Abstracts**: The COVID-19 pandemic has impacted our society by forcing shutdowns and shifting the way people interacted worldwide. In relation to the impacts on the electric grid, it created a significant decrease in energy demands across the globe. Recent studies have shown that the low demand conditions caused by COVID-19 lockdowns combined with large renewable generation have resulted in extremely low-inertia grid conditions. In this work, we examine how an attacker could exploit these {scenarios} to cause unsafe grid operating conditions by executing load-altering attacks (LAAs) targeted at compromising hundreds of thousands of IoT-connected high-wattage loads in low-inertia power systems. Our study focuses on analyzing the impact of the COVID-19 mitigation measures on U.S. regional transmission operators (RTOs), formulating a plausible and realistic least-effort LAA targeted at transmission systems with low-inertia conditions, and evaluating the probability of these large-scale LAAs. Theoretical and simulation results are presented based on the WSCC 9-bus {and IEEE 118-bus} test systems. Results demonstrate how adversaries could provoke major frequency disturbances by targeting vulnerable load buses in low-inertia systems and offer insights into how the temporal fluctuations of renewable energy sources, considering generation scheduling, impact the grid's vulnerability to LAAs.



## **42. MaMaDroid2.0 -- The Holes of Control Flow Graphs**

cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13922v1)

**Authors**: Harel Berger, Chen Hajaj, Enrico Mariconti, Amit Dvir

**Abstracts**: Android malware is a continuously expanding threat to billions of mobile users around the globe. Detection systems are updated constantly to address these threats. However, a backlash takes the form of evasion attacks, in which an adversary changes malicious samples such that those samples will be misclassified as benign. This paper fully inspects a well-known Android malware detection system, MaMaDroid, which analyzes the control flow graph of the application. Changes to the portion of benign samples in the train set and models are considered to see their effect on the classifier. The changes in the ratio between benign and malicious samples have a clear effect on each one of the models, resulting in a decrease of more than 40% in their detection rate. Moreover, adopted ML models are implemented as well, including 5-NN, Decision Tree, and Adaboost. Exploration of the six models reveals a typical behavior in different cases, of tree-based models and distance-based models. Moreover, three novel attacks that manipulate the CFG and their detection rates are described for each one of the targeted models. The attacks decrease the detection rate of most of the models to 0%, with regards to different ratios of benign to malicious apps. As a result, a new version of MaMaDroid is engineered. This model fuses the CFG of the app and static analysis of features of the app. This improved model is proved to be robust against evasion attacks targeting both CFG-based models and static analysis models, achieving a detection rate of more than 90% against each one of the attacks.



## **43. Formally verified asymptotic consensus in robust networks**

cs.PL

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13833v1)

**Authors**: Mohit Tekriwal, Avi Tachna-Fram, Jean-Baptiste Jeannin, Manos Kapritsos, Dimitra Panagou

**Abstracts**: Distributed architectures are used to improve performance and reliability of various systems. An important capability of a distributed architecture is the ability to reach consensus among all its nodes. To achieve this, several consensus algorithms have been proposed for various scenarii, and many of these algorithms come with proofs of correctness that are not mechanically checked. Unfortunately, those proofs are known to be intricate and prone to errors.   In this paper, we formalize and mechanically check a consensus algorithm widely used in the distributed controls community: the Weighted-Mean Subsequence Reduced (W-MSR) algorithm proposed by Le Blanc et al. This algorithm provides a way to achieve asymptotic consensus in a distributed controls scenario in the presence of adversarial agents (attackers) that may not update their states based on the nominal consensus protocol, and may share inaccurate information with their neighbors. Using the Coq proof assistant, we formalize the necessary and sufficient conditions required to achieve resilient asymptotic consensus under the assumed attacker model. We leverage the existing Coq formalizations of graph theory, finite sets and sequences of the mathcomp library for our development. To our knowledge, this is the first mechanical proof of an asymptotic consensus algorithm. During the formalization, we clarify several imprecisions in the paper proof, including an imprecision on quantifiers in the main theorem.



## **44. Robust Textual Embedding against Word-level Adversarial Attacks**

cs.CL

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13817v1)

**Authors**: Yichen Yang, Xiaosen Wang, Kun He

**Abstracts**: We attribute the vulnerability of natural language processing models to the fact that similar inputs are converted to dissimilar representations in the embedding space, leading to inconsistent outputs, and propose a novel robust training method, termed Fast Triplet Metric Learning (FTML). Specifically, we argue that the original sample should have similar representation with its adversarial counterparts and distinguish its representation from other samples for better robustness. To this end, we adopt the triplet metric learning into the standard training to pull the words closer to their positive samples (i.e., synonyms) and push away their negative samples (i.e., non-synonyms) in the embedding space. Extensive experiments demonstrate that FTML can significantly promote the model robustness against various advanced adversarial attacks while keeping competitive classification accuracy on original samples. Besides, our method is efficient as it only needs to adjust the embedding and introduces very little overhead on the standard training. Our work shows the great potential of improving the textual robustness through robust word embedding.



## **45. On the Robustness of CountSketch to Adaptive Inputs**

cs.DS

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13736v1)

**Authors**: Edith Cohen, Xin Lyu, Jelani Nelson, Tamás Sarlós, Moshe Shechner, Uri Stemmer

**Abstracts**: CountSketch is a popular dimensionality reduction technique that maps vectors to a lower dimension using randomized linear measurements. The sketch supports recovering $\ell_2$-heavy hitters of a vector (entries with $v[i]^2 \geq \frac{1}{k}\|\boldsymbol{v}\|^2_2$). We study the robustness of the sketch in adaptive settings where input vectors may depend on the output from prior inputs. Adaptive settings arise in processes with feedback or with adversarial attacks. We show that the classic estimator is not robust, and can be attacked with a number of queries of the order of the sketch size. We propose a robust estimator (for a slightly modified sketch) that allows for quadratic number of queries in the sketch size, which is an improvement factor of $\sqrt{k}$ (for $k$ heavy hitters) over prior work.



## **46. An Empirical Study on the Intrinsic Privacy of SGD**

cs.LG

21 pages, 11 figures, 8 tables

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/1912.02919v4)

**Authors**: Stephanie L. Hyland, Shruti Tople

**Abstracts**: Introducing noise in the training of machine learning systems is a powerful way to protect individual privacy via differential privacy guarantees, but comes at a cost to utility. This work looks at whether the inherent randomness of stochastic gradient descent (SGD) could contribute to privacy, effectively reducing the amount of \emph{additional} noise required to achieve a given privacy guarantee. We conduct a large-scale empirical study to examine this question. Training a grid of over 120,000 models across four datasets (tabular and images) on convex and non-convex objectives, we demonstrate that the random seed has a larger impact on model weights than any individual training example. We test the distribution over weights induced by the seed, finding that the simple convex case can be modelled with a multivariate Gaussian posterior, while neural networks exhibit multi-modal and non-Gaussian weight distributions. By casting convex SGD as a Gaussian mechanism, we then estimate an `intrinsic' data-dependent $\epsilon_i(\mathcal{D})$, finding values as low as 6.3, dropping to 1.9 using empirical estimates. We use a membership inference attack to estimate $\epsilon$ for non-convex SGD and demonstrate that hiding the random seed from the adversary results in a statistically significant reduction in attack performance, corresponding to a reduction in the effective $\epsilon$. These results provide empirical evidence that SGD exhibits appreciable variability relative to its dataset sensitivity, and this `intrinsic noise' has the potential to be leveraged to improve the utility of privacy-preserving machine learning.



## **47. Enhance transferability of adversarial examples with model architecture**

cs.LG

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13625v1)

**Authors**: Mingyuan Fan, Wenzhong Guo, Shengxing Yu, Zuobin Ying, Ximeng Liu

**Abstracts**: Transferability of adversarial examples is of critical importance to launch black-box adversarial attacks, where attackers are only allowed to access the output of the target model. However, under such a challenging but practical setting, the crafted adversarial examples are always prone to overfitting to the proxy model employed, presenting poor transferability. In this paper, we suggest alleviating the overfitting issue from a novel perspective, i.e., designing a fitted model architecture. Specifically, delving the bottom of the cause of poor transferability, we arguably decompose and reconstruct the existing model architecture into an effective model architecture, namely multi-track model architecture (MMA). The adversarial examples crafted on the MMA can maximumly relieve the effect of model-specified features to it and toward the vulnerable directions adopted by diverse architectures. Extensive experimental evaluation demonstrates that the transferability of adversarial examples based on the MMA significantly surpass other state-of-the-art model architectures by up to 40% with comparable overhead.



## **48. GRAPHITE: Generating Automatic Physical Examples for Machine-Learning Attacks on Computer Vision Systems**

cs.CR

IEEE European Symposium on Security and Privacy 2022 (EuroS&P 2022)

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2002.07088v6)

**Authors**: Ryan Feng, Neal Mangaokar, Jiefeng Chen, Earlence Fernandes, Somesh Jha, Atul Prakash

**Abstracts**: This paper investigates an adversary's ease of attack in generating adversarial examples for real-world scenarios. We address three key requirements for practical attacks for the real-world: 1) automatically constraining the size and shape of the attack so it can be applied with stickers, 2) transform-robustness, i.e., robustness of a attack to environmental physical variations such as viewpoint and lighting changes, and 3) supporting attacks in not only white-box, but also black-box hard-label scenarios, so that the adversary can attack proprietary models. In this work, we propose GRAPHITE, an efficient and general framework for generating attacks that satisfy the above three key requirements. GRAPHITE takes advantage of transform-robustness, a metric based on expectation over transforms (EoT), to automatically generate small masks and optimize with gradient-free optimization. GRAPHITE is also flexible as it can easily trade-off transform-robustness, perturbation size, and query count in black-box settings. On a GTSRB model in a hard-label black-box setting, we are able to find attacks on all possible 1,806 victim-target class pairs with averages of 77.8% transform-robustness, perturbation size of 16.63% of the victim images, and 126K queries per pair. For digital-only attacks where achieving transform-robustness is not a requirement, GRAPHITE is able to find successful small-patch attacks with an average of only 566 queries for 92.2% of victim-target pairs. GRAPHITE is also able to find successful attacks using perturbations that modify small areas of the input image against PatchGuard, a recently proposed defense against patch-based attacks.



## **49. Markov Chain Monte Carlo-Based Machine Unlearning: Unlearning What Needs to be Forgotten**

cs.LG

Proceedings of the 2022 ACM Asia Conference on Computer and  Communications Security (ASIA CCS '22), May 30-June 3, 2022, Nagasaki, Japan

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13585v1)

**Authors**: Quoc Phong Nguyen, Ryutaro Oikawa, Dinil Mon Divakaran, Mun Choon Chan, Bryan Kian Hsiang Low

**Abstracts**: As the use of machine learning (ML) models is becoming increasingly popular in many real-world applications, there are practical challenges that need to be addressed for model maintenance. One such challenge is to 'undo' the effect of a specific subset of dataset used for training a model. This specific subset may contain malicious or adversarial data injected by an attacker, which affects the model performance. Another reason may be the need for a service provider to remove data pertaining to a specific user to respect the user's privacy. In both cases, the problem is to 'unlearn' a specific subset of the training data from a trained model without incurring the costly procedure of retraining the whole model from scratch. Towards this goal, this paper presents a Markov chain Monte Carlo-based machine unlearning (MCU) algorithm. MCU helps to effectively and efficiently unlearn a trained model from subsets of training dataset. Furthermore, we show that with MCU, we are able to explain the effect of a subset of a training dataset on the model prediction. Thus, MCU is useful for examining subsets of data to identify the adversarial data to be removed. Similarly, MCU can be used to erase the lineage of a user's personal data from trained ML models, thus upholding a user's "right to be forgotten". We empirically evaluate the performance of our proposed MCU algorithm on real-world phishing and diabetes datasets. Results show that MCU can achieve a desirable performance by efficiently removing the effect of a subset of training dataset and outperform an existing algorithm that utilizes the remaining dataset.



## **50. A Unified Wasserstein Distributional Robustness Framework for Adversarial Training**

cs.LG

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2202.13437v1)

**Authors**: Tuan Anh Bui, Trung Le, Quan Tran, He Zhao, Dinh Phung

**Abstracts**: It is well-known that deep neural networks (DNNs) are susceptible to adversarial attacks, exposing a severe fragility of deep learning systems. As the result, adversarial training (AT) method, by incorporating adversarial examples during training, represents a natural and effective approach to strengthen the robustness of a DNN-based classifier. However, most AT-based methods, notably PGD-AT and TRADES, typically seek a pointwise adversary that generates the worst-case adversarial example by independently perturbing each data sample, as a way to "probe" the vulnerability of the classifier. Arguably, there are unexplored benefits in considering such adversarial effects from an entire distribution. To this end, this paper presents a unified framework that connects Wasserstein distributional robustness with current state-of-the-art AT methods. We introduce a new Wasserstein cost function and a new series of risk functions, with which we show that standard AT methods are special cases of their counterparts in our framework. This connection leads to an intuitive relaxation and generalization of existing AT methods and facilitates the development of a new family of distributional robustness AT-based algorithms. Extensive experiments show that our distributional robustness AT algorithms robustify further their standard AT counterparts in various settings.



