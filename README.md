# Latest Adversarial Attack Papers
**update at 2022-06-10 06:31:31**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. PrivHAR: Recognizing Human Actions From Privacy-preserving Lens**

cs.CV

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03891v1)

**Authors**: Carlos Hinojosa, Miguel Marquez, Henry Arguello, Ehsan Adeli, Li Fei-Fei, Juan Carlos Niebles

**Abstracts**: The accelerated use of digital cameras prompts an increasing concern about privacy and security, particularly in applications such as action recognition. In this paper, we propose an optimizing framework to provide robust visual privacy protection along the human action recognition pipeline. Our framework parameterizes the camera lens to successfully degrade the quality of the videos to inhibit privacy attributes and protect against adversarial attacks while maintaining relevant features for activity recognition. We validate our approach with extensive simulations and hardware experiments.



## **2. Standalone Neural ODEs with Sensitivity Analysis**

cs.LG

25 pages, 15 figures; typos corrected

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2205.13933v2)

**Authors**: Rym Jaroudi, Lukáš Malý, Gabriel Eilertsen, B. Tomas Johansson, Jonas Unger, George Baravdish

**Abstracts**: This paper presents the Standalone Neural ODE (sNODE), a continuous-depth neural ODE model capable of describing a full deep neural network. This uses a novel nonlinear conjugate gradient (NCG) descent optimization scheme for training, where the Sobolev gradient can be incorporated to improve smoothness of model weights. We also present a general formulation of the neural sensitivity problem and show how it is used in the NCG training. The sensitivity analysis provides a reliable measure of uncertainty propagation throughout a network, and can be used to study model robustness and to generate adversarial attacks. Our evaluations demonstrate that our novel formulations lead to increased robustness and performance as compared to ResNet models, and that it opens up for new opportunities for designing and developing machine learning with improved explainability.



## **3. Wavelet Regularization Benefits Adversarial Training**

cs.CV

Preprint version

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03727v1)

**Authors**: Jun Yan, Huilin Yin, Xiaoyang Deng, Ziming Zhao, Wancheng Ge, Hao Zhang, Gerhard Rigoll

**Abstracts**: Adversarial training methods are state-of-the-art (SOTA) empirical defense methods against adversarial examples. Many regularization methods have been proven to be effective with the combination of adversarial training. Nevertheless, such regularization methods are implemented in the time domain. Since adversarial vulnerability can be regarded as a high-frequency phenomenon, it is essential to regulate the adversarially-trained neural network models in the frequency domain. Faced with these challenges, we make a theoretical analysis on the regularization property of wavelets which can enhance adversarial training. We propose a wavelet regularization method based on the Haar wavelet decomposition which is named Wavelet Average Pooling. This wavelet regularization module is integrated into the wide residual neural network so that a new WideWaveletResNet model is formed. On the datasets of CIFAR-10 and CIFAR-100, our proposed Adversarial Wavelet Training method realizes considerable robustness under different types of attacks. It verifies the assumption that our wavelet regularization method can enhance adversarial robustness especially in the deep wide neural networks. The visualization experiments of the Frequency Principle (F-Principle) and interpretability are implemented to show the effectiveness of our method. A detailed comparison based on different wavelet base functions is presented. The code is available at the repository: \url{https://github.com/momo1986/AdversarialWaveletTraining}.



## **4. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

cs.IR

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2204.01321v3)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed. In this paper, we introduce the Word Substitution Ranking Attack (WSRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers cannot directly get access to the model information, but can only query the target model to obtain the rank positions of the partial retrieved list. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations. Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.



## **5. Latent Boundary-guided Adversarial Training**

cs.LG

To appear in Machine Learning

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03717v1)

**Authors**: Xiaowei Zhou, Ivor W. Tsang, Jie Yin

**Abstracts**: Deep Neural Networks (DNNs) have recently achieved great success in many classification tasks. Unfortunately, they are vulnerable to adversarial attacks that generate adversarial examples with a small perturbation to fool DNN models, especially in model sharing scenarios. Adversarial training is proved to be the most effective strategy that injects adversarial examples into model training to improve the robustness of DNN models to adversarial attacks. However, adversarial training based on the existing adversarial examples fails to generalize well to standard, unperturbed test data. To achieve a better trade-off between standard accuracy and adversarial robustness, we propose a novel adversarial training framework called LAtent bounDary-guided aDvErsarial tRaining (LADDER) that adversarially trains DNN models on latent boundary-guided adversarial examples. As opposed to most of the existing methods that generate adversarial examples in the input space, LADDER generates a myriad of high-quality adversarial examples through adding perturbations to latent features. The perturbations are made along the normal of the decision boundary constructed by an SVM with an attention mechanism. We analyze the merits of our generated boundary-guided adversarial examples from a boundary field perspective and visualization view. Extensive experiments and detailed analysis on MNIST, SVHN, CelebA, and CIFAR-10 validate the effectiveness of LADDER in achieving a better trade-off between standard accuracy and adversarial robustness as compared with vanilla DNNs and competitive baselines.



## **6. Autoregressive Perturbations for Data Poisoning**

cs.LG

21 pages, 13 figures. Code available at  https://github.com/psandovalsegura/autoregressive-poisoning

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03693v1)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs

**Abstracts**: The prevalence of data scraping from social media as a means to obtain datasets has led to growing concerns regarding unauthorized use of data. Data poisoning attacks have been proposed as a bulwark against scraping, as they make data "unlearnable" by adding small, imperceptible perturbations. Unfortunately, existing methods require knowledge of both the target architecture and the complete dataset so that a surrogate network can be trained, the parameters of which are used to generate the attack. In this work, we introduce autoregressive (AR) poisoning, a method that can generate poisoned data without access to the broader dataset. The proposed AR perturbations are generic, can be applied across different datasets, and can poison different architectures. Compared to existing unlearnable methods, our AR poisons are more resistant against common defenses such as adversarial training and strong data augmentations. Our analysis further provides insight into what makes an effective data poison.



## **7. SHORTSTACK: Distributed, Fault-tolerant, Oblivious Data Access**

cs.CR

Full version of USENIX OSDI'22 paper

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2205.14281v2)

**Authors**: Midhul Vuppalapati, Kushal Babel, Anurag Khandelwal, Rachit Agarwal

**Abstracts**: Many applications that benefit from data offload to cloud services operate on private data. A now-long line of work has shown that, even when data is offloaded in an encrypted form, an adversary can learn sensitive information by analyzing data access patterns. Existing techniques for oblivious data access-that protect against access pattern attacks-require a centralized and stateful trusted proxy to orchestrate data accesses from applications to cloud services. We show that, in failure-prone deployments, such a centralized and stateful proxy results in violation of oblivious data access security guarantees and/or system unavailability. We thus initiate the study of distributed, fault-tolerant, oblivious data access.   We present SHORTSTACK, a distributed proxy architecture for oblivious data access in failure-prone deployments. SHORTSTACK achieves the classical obliviousness guarantee--access patterns observed by the adversary being independent of the input--even under a powerful passive persistent adversary that can force failure of arbitrary (bounded-sized) subset of proxy servers at arbitrary times. We also introduce a security model that enables studying oblivious data access with distributed, failure-prone, servers. We provide a formal proof that SHORTSTACK enables oblivious data access under this model, and show empirically that SHORTSTACK performance scales near-linearly with number of distributed proxy servers.



## **8. Dap-FL: Federated Learning flourishes by adaptive tuning and secure aggregation**

cs.CR

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03623v1)

**Authors**: Qian Chen, Zilong Wang, Jiawei Chen, Haonan Yan, Xiaodong Lin

**Abstracts**: Federated learning (FL), an attractive and promising distributed machine learning paradigm, has sparked extensive interest in exploiting tremendous data stored on ubiquitous mobile devices. However, conventional FL suffers severely from resource heterogeneity, as clients with weak computational and communication capability may be unable to complete local training using the same local training hyper-parameters. In this paper, we propose Dap-FL, a deep deterministic policy gradient (DDPG)-assisted adaptive FL system, in which local learning rates and local training epochs are adaptively adjusted by all resource-heterogeneous clients through locally deployed DDPG-assisted adaptive hyper-parameter selection schemes. Particularly, the rationality of the proposed hyper-parameter selection scheme is confirmed through rigorous mathematical proof. Besides, due to the thoughtlessness of security consideration of adaptive FL systems in previous studies, we introduce the Paillier cryptosystem to aggregate local models in a secure and privacy-preserving manner. Rigorous analyses show that the proposed Dap-FL system could guarantee the security of clients' private local models against chosen-plaintext attacks and chosen-message attacks in a widely used honest-but-curious participants and active adversaries security model. In addition, through ingenious and extensive experiments, the proposed Dap-FL achieves higher global model prediction accuracy and faster convergence rates than conventional FL, and the comprehensiveness of the adjusted local training hyper-parameters is validated. More importantly, experimental results also show that the proposed Dap-FL achieves higher model prediction accuracy than two state-of-the-art RL-assisted FL methods, i.e., 6.03% higher than DDPG-based FL and 7.85% higher than DQN-based FL.



## **9. Blacklight: Defending Black-Box Adversarial Attacks on Deep Neural Networks**

cs.CR

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2006.14042v2)

**Authors**: Huiying Li, Shawn Shan, Emily Wenger, Jiayun Zhang, Haitao Zheng, Ben Y. Zhao

**Abstracts**: Deep learning systems are known to be vulnerable to adversarial examples. In particular, query-based black-box attacks do not require knowledge of the deep learning model, but can compute adversarial examples over the network by submitting queries and inspecting returns. Recent work largely improves the efficiency of those attacks, demonstrating their practicality on today's ML-as-a-service platforms.   We propose Blacklight, a new defense against query-based black-box adversarial attacks. The fundamental insight driving our design is that, to compute adversarial examples, these attacks perform iterative optimization over the network, producing image queries highly similar in the input space. Blacklight detects query-based black-box attacks by detecting highly similar queries, using an efficient similarity engine operating on probabilistic content fingerprints. We evaluate Blacklight against eight state-of-the-art attacks, across a variety of models and image classification tasks. Blacklight identifies them all, often after only a handful of queries. By rejecting all detected queries, Blacklight prevents any attack to complete, even when attackers persist to submit queries after account ban or query rejection. Blacklight is also robust against several powerful countermeasures, including an optimal black-box attack that approximates white-box attacks in efficiency. Finally, we illustrate how Blacklight generalizes to other domains like text classification.



## **10. Random and Adversarial Bit Error Robustness: Energy-Efficient and Secure DNN Accelerators**

cs.LG

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2104.08323v2)

**Authors**: David Stutz, Nandhini Chandramoorthy, Matthias Hein, Bernt Schiele

**Abstracts**: Deep neural network (DNN) accelerators received considerable attention in recent years due to the potential to save energy compared to mainstream hardware. Low-voltage operation of DNN accelerators allows to further reduce energy consumption, however, causes bit-level failures in the memory storing the quantized weights. Furthermore, DNN accelerators are vulnerable to adversarial attacks on voltage controllers or individual bits. In this paper, we show that a combination of robust fixed-point quantization, weight clipping, as well as random bit error training (RandBET) or adversarial bit error training (AdvBET) improves robustness against random or adversarial bit errors in quantized DNN weights significantly. This leads not only to high energy savings for low-voltage operation as well as low-precision quantization, but also improves security of DNN accelerators. In contrast to related work, our approach generalizes across operating voltages and accelerators and does not require hardware changes. Moreover, we present a novel adversarial bit error attack and are able to obtain robustness against both targeted and untargeted bit-level attacks. Without losing more than 0.8%/2% in test accuracy, we can reduce energy consumption on CIFAR10 by 20%/30% for 8/4-bit quantization. Allowing up to 320 adversarial bit errors, we reduce test error from above 90% (chance level) to 26.22%.



## **11. Optimal Clock Synchronization with Signatures**

cs.DC

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2203.02553v2)

**Authors**: Christoph Lenzen, Julian Loss

**Abstracts**: Cryptographic signatures can be used to increase the resilience of distributed systems against adversarial attacks, by increasing the number of faulty parties that can be tolerated. While this is well-studied for consensus, it has been underexplored in the context of fault-tolerant clock synchronization, even in fully connected systems. Here, the honest parties of an $n$-node system are required to compute output clocks of small skew (i.e., maximum phase offset) despite local clock rates varying between $1$ and $\vartheta>1$, end-to-end communication delays varying between $d-u$ and $d$, and the interference from malicious parties. So far, it is only known that clock pulses of skew $d$ can be generated with (trivially optimal) resilience of $\lceil n/2\rceil-1$ (PODC `19), improving over the tight bound of $\lceil n/3\rceil-1$ holding without signatures for \emph{any} skew bound (STOC `84, PODC `85). Since typically $d\gg u$ and $\vartheta-1\ll 1$, this is far from the lower bound of $u+(\vartheta-1)d$ that applies even in the fault-free case (IPL `01).   We prove matching upper and lower bounds of $\Theta(u+(\vartheta-1)d)$ on the skew for the resilience range from $\lceil n/3\rceil$ to $\lceil n/2\rceil-1$. The algorithm showing the upper bound is, under the assumption that the adversary cannot forge signatures, deterministic. The lower bound holds even if clocks are initially perfectly synchronized, message delays between honest nodes are known, $\vartheta$ is arbitrarily close to one, and the synchronization algorithm is randomized. This has crucial implications for network designers that seek to leverage signatures for providing more robust time. In contrast to the setting without signatures, they must ensure that an attacker cannot easily bypass the lower bound on the delay on links with a faulty endpoint.



## **12. Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks**

cs.LG

Accepted by ICML 2022 as Oral

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2201.12179v3)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Antonio De Almeida Correia, Antonia Adler, Kristian Kersting

**Abstracts**: Model inversion attacks (MIAs) aim to create synthetic images that reflect the class-wise characteristics from a target classifier's private training data by exploiting the model's learned knowledge. Previous research has developed generative MIAs that use generative adversarial networks (GANs) as image priors tailored to a specific target model. This makes the attacks time- and resource-consuming, inflexible, and susceptible to distributional shifts between datasets. To overcome these drawbacks, we present Plug & Play Attacks, which relax the dependency between the target model and image prior, and enable the use of a single GAN to attack a wide range of targets, requiring only minor adjustments to the attack. Moreover, we show that powerful MIAs are possible even with publicly available pre-trained GANs and under strong distributional shifts, for which previous approaches fail to produce meaningful results. Our extensive evaluation confirms the improved robustness and flexibility of Plug & Play Attacks and their ability to create high-quality images revealing sensitive class characteristics.



## **13. Towards Understanding and Mitigating Audio Adversarial Examples for Speaker Recognition**

cs.SD

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03393v1)

**Authors**: Guangke Chen, Zhe Zhao, Fu Song, Sen Chen, Lingling Fan, Feng Wang, Jiashui Wang

**Abstracts**: Speaker recognition systems (SRSs) have recently been shown to be vulnerable to adversarial attacks, raising significant security concerns. In this work, we systematically investigate transformation and adversarial training based defenses for securing SRSs. According to the characteristic of SRSs, we present 22 diverse transformations and thoroughly evaluate them using 7 recent promising adversarial attacks (4 white-box and 3 black-box) on speaker recognition. With careful regard for best practices in defense evaluations, we analyze the strength of transformations to withstand adaptive attacks. We also evaluate and understand their effectiveness against adaptive attacks when combined with adversarial training. Our study provides lots of useful insights and findings, many of them are new or inconsistent with the conclusions in the image and speech recognition domains, e.g., variable and constant bit rate speech compressions have different performance, and some non-differentiable transformations remain effective against current promising evasion techniques which often work well in the image domain. We demonstrate that the proposed novel feature-level transformation combined with adversarial training is rather effective compared to the sole adversarial training in a complete white-box setting, e.g., increasing the accuracy by 13.62% and attack cost by two orders of magnitude, while other transformations do not necessarily improve the overall defense capability. This work sheds further light on the research directions in this field. We also release our evaluation platform SPEAKERGUARD to foster further research.



## **14. Building Robust Ensembles via Margin Boosting**

cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03362v1)

**Authors**: Dinghuai Zhang, Hongyang Zhang, Aaron Courville, Yoshua Bengio, Pradeep Ravikumar, Arun Sai Suggala

**Abstracts**: In the context of adversarial robustness, a single model does not usually have enough power to defend against all possible adversarial attacks, and as a result, has sub-optimal robustness. Consequently, an emerging line of work has focused on learning an ensemble of neural networks to defend against adversarial attacks. In this work, we take a principled approach towards building robust ensembles. We view this problem from the perspective of margin-boosting and develop an algorithm for learning an ensemble with maximum margin. Through extensive empirical evaluation on benchmark datasets, we show that our algorithm not only outperforms existing ensembling techniques, but also large models trained in an end-to-end fashion. An important byproduct of our work is a margin-maximizing cross-entropy (MCE) loss, which is a better alternative to the standard cross-entropy (CE) loss. Empirically, we show that replacing the CE loss in state-of-the-art adversarial training techniques with our MCE loss leads to significant performance improvement.



## **15. Adaptive Regularization for Adversarial Training**

stat.ML

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03353v1)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstracts**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to use a data-adaptive regularization for robustifying a prediction model. We apply more regularization to data which are more vulnerable to adversarial attacks and vice versa. Even though the idea of data-adaptive regularization is not new, our data-adaptive regularization has a firm theoretical base of reducing an upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on clean samples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.



## **16. AS2T: Arbitrary Source-To-Target Adversarial Attack on Speaker Recognition Systems**

cs.SD

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03351v1)

**Authors**: Guangke Chen, Zhe Zhao, Fu Song, Sen Chen, Lingling Fan, Yang Liu

**Abstracts**: Recent work has illuminated the vulnerability of speaker recognition systems (SRSs) against adversarial attacks, raising significant security concerns in deploying SRSs. However, they considered only a few settings (e.g., some combinations of source and target speakers), leaving many interesting and important settings in real-world attack scenarios alone. In this work, we present AS2T, the first attack in this domain which covers all the settings, thus allows the adversary to craft adversarial voices using arbitrary source and target speakers for any of three main recognition tasks. Since none of the existing loss functions can be applied to all the settings, we explore many candidate loss functions for each setting including the existing and newly designed ones. We thoroughly evaluate their efficacy and find that some existing loss functions are suboptimal. Then, to improve the robustness of AS2T towards practical over-the-air attack, we study the possible distortions occurred in over-the-air transmission, utilize different transformation functions with different parameters to model those distortions, and incorporate them into the generation of adversarial voices. Our simulated over-the-air evaluation validates the effectiveness of our solution in producing robust adversarial voices which remain effective under various hardware devices and various acoustic environments with different reverberation, ambient noises, and noise levels. Finally, we leverage AS2T to perform thus far the largest-scale evaluation to understand transferability among 14 diverse SRSs. The transferability analysis provides many interesting and useful insights which challenge several findings and conclusion drawn in previous works in the image domain. Our study also sheds light on future directions of adversarial attacks in the speaker recognition domain.



## **17. Subject Membership Inference Attacks in Federated Learning**

cs.LG

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03317v1)

**Authors**: Anshuman Suri, Pallika Kanani, Virendra J. Marathe, Daniel W. Peterson

**Abstracts**: Privacy in Federated Learning (FL) is studied at two different granularities: item-level, which protects individual data points, and user-level, which protects each user (participant) in the federation. Nearly all of the private FL literature is dedicated to studying privacy attacks and defenses at these two granularities. Recently, subject-level privacy has emerged as an alternative privacy granularity to protect the privacy of individuals (data subjects) whose data is spread across multiple (organizational) users in cross-silo FL settings. An adversary might be interested in recovering private information about these individuals (a.k.a. \emph{data subjects}) by attacking the trained model. A systematic study of these patterns requires complete control over the federation, which is impossible with real-world datasets. We design a simulator for generating various synthetic federation configurations, enabling us to study how properties of the data, model design and training, and the federation itself impact subject privacy risk. We propose three attacks for \emph{subject membership inference} and examine the interplay between all factors within a federation that affect the attacks' efficacy. We also investigate the effectiveness of Differential Privacy in mitigating this threat. Our takeaways generalize to real-world datasets like FEMNIST, giving credence to our findings.



## **18. Quickest Change Detection in the Presence of Transient Adversarial Attacks**

eess.SP

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03245v1)

**Authors**: Thirupathaiah Vasantam, Don Towsley, Venugopal V. Veeravalli

**Abstracts**: We study a monitoring system in which the distributions of sensors' observations change from a nominal distribution to an abnormal distribution in response to an adversary's presence. The system uses the quickest change detection procedure, the Shewhart rule, to detect the adversary that uses its resources to affect the abnormal distribution, so as to hide its presence. The metric of interest is the probability of missed detection within a predefined number of time-slots after the changepoint. Assuming that the adversary's resource constraints are known to the detector, we find the number of required sensors to make the worst-case probability of missed detection less than an acceptable level. The distributions of observations are assumed to be Gaussian, and the presence of the adversary affects their mean. We also provide simulation results to support our analysis.



## **19. Robust Adversarial Attacks Detection based on Explainable Deep Reinforcement Learning For UAV Guidance and Planning**

cs.LG

13 pages, 20 figures

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.02670v2)

**Authors**: Thomas Hickling, Nabil Aouf, Phillippa Spencer

**Abstracts**: The danger of adversarial attacks to unprotected Uncrewed Aerial Vehicle (UAV) agents operating in public is growing. Adopting AI-based techniques and more specifically Deep Learning (DL) approaches to control and guide these UAVs can be beneficial in terms of performance but add more concerns regarding the safety of those techniques and their vulnerability against adversarial attacks causing the chances of collisions going up as the agent becomes confused. This paper proposes an innovative approach based on the explainability of DL methods to build an efficient detector that will protect these DL schemes and thus the UAVs adopting them from potential attacks. The agent is adopting a Deep Reinforcement Learning (DRL) scheme for guidance and planning. It is formed and trained with a Deep Deterministic Policy Gradient (DDPG) with Prioritised Experience Replay (PER) DRL scheme that utilises Artificial Potential Field (APF) to improve training times and obstacle avoidance performance. The adversarial attacks are generated by Fast Gradient Sign Method (FGSM) and Basic Iterative Method (BIM) algorithms and reduced obstacle course completion rates from 80\% to 35\%. A Realistic Synthetic environment for UAV explainable DRL based planning and guidance including obstacles and adversarial attacks is built. Two adversarial attack detectors are proposed. The first one adopts a Convolutional Neural Network (CNN) architecture and achieves an accuracy in detection of 80\%. The second detector is developed based on a Long Short Term Memory (LSTM) network and achieves an accuracy of 91\% with much faster computing times when compared to the CNN based detector.



## **20. VLC Physical Layer Security through RIS-aided Jamming Receiver for 6G Wireless Networks**

cs.CR

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2205.09026v2)

**Authors**: Simone Soderi, Alessandro Brighente, Federico Turrin, Mauro Conti

**Abstracts**: Visible Light Communication (VLC) is one the most promising enabling technology for future 6G networks to overcome Radio-Frequency (RF)-based communication limitations thanks to a broader bandwidth, higher data rate, and greater efficiency. However, from the security perspective, VLCs suffer from all known wireless communication security threats (e.g., eavesdropping and integrity attacks). For this reason, security researchers are proposing innovative Physical Layer Security (PLS) solutions to protect such communication. Among the different solutions, the novel Reflective Intelligent Surface (RIS) technology coupled with VLCs has been successfully demonstrated in recent work to improve the VLC communication capacity. However, to date, the literature still lacks analysis and solutions to show the PLS capability of RIS-based VLC communication. In this paper, we combine watermarking and jamming primitives through the Watermark Blind Physical Layer Security (WBPLSec) algorithm to secure VLC communication at the physical layer. Our solution leverages RIS technology to improve the security properties of the communication. By using an optimization framework, we can calculate RIS phases to maximize the WBPLSec jamming interference schema over a predefined area in the room. In particular, compared to a scenario without RIS, our solution improves the performance in terms of secrecy capacity without any assumption about the adversary's location. We validate through numerical evaluations the positive impact of RIS-aided solution to increase the secrecy capacity of the legitimate jamming receiver in a VLC indoor scenario. Our results show that the introduction of RIS technology extends the area where secure communication occurs and that by increasing the number of RIS elements the outage probability decreases.



## **21. Sampling without Replacement Leads to Faster Rates in Finite-Sum Minimax Optimization**

math.OC

48 pages, 3 figures

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.02953v1)

**Authors**: Aniket Das, Bernhard Schölkopf, Michael Muehlebach

**Abstracts**: We analyze the convergence rates of stochastic gradient algorithms for smooth finite-sum minimax optimization and show that, for many such algorithms, sampling the data points without replacement leads to faster convergence compared to sampling with replacement. For the smooth and strongly convex-strongly concave setting, we consider gradient descent ascent and the proximal point method, and present a unified analysis of two popular without-replacement sampling strategies, namely Random Reshuffling (RR), which shuffles the data every epoch, and Single Shuffling or Shuffle Once (SO), which shuffles only at the beginning. We obtain tight convergence rates for RR and SO and demonstrate that these strategies lead to faster convergence than uniform sampling. Moving beyond convexity, we obtain similar results for smooth nonconvex-nonconcave objectives satisfying a two-sided Polyak-{\L}ojasiewicz inequality. Finally, we demonstrate that our techniques are general enough to analyze the effect of data-ordering attacks, where an adversary manipulates the order in which data points are supplied to the optimizer. Our analysis also recovers tight rates for the incremental gradient method, where the data points are not shuffled at all.



## **22. A Robust Deep Learning Enabled Semantic Communication System for Text**

eess.SP

6 pages

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2206.02596v1)

**Authors**: Xiang Peng, Zhijin Qin, Danlan Huang, Xiaoming Tao, Jianhua Lu, Guangyi Liu, Chengkang Pan

**Abstracts**: With the advent of the 6G era, the concept of semantic communication has attracted increasing attention. Compared with conventional communication systems, semantic communication systems are not only affected by physical noise existing in the wireless communication environment, e.g., additional white Gaussian noise, but also by semantic noise due to the source and the nature of deep learning-based systems. In this paper, we elaborate on the mechanism of semantic noise. In particular, we categorize semantic noise into two categories: literal semantic noise and adversarial semantic noise. The former is caused by written errors or expression ambiguity, while the latter is caused by perturbations or attacks added to the embedding layer via the semantic channel. To prevent semantic noise from influencing semantic communication systems, we present a robust deep learning enabled semantic communication system (R-DeepSC) that leverages a calibrated self-attention mechanism and adversarial training to tackle semantic noise. Compared with baseline models that only consider physical noise for text transmission, the proposed R-DeepSC achieves remarkable performance in dealing with semantic noise under different signal-to-noise ratios.



## **23. Certified Robustness in Federated Learning**

cs.LG

17 pages, 10 figures. Code available at  https://github.com/MotasemAlfarra/federated-learning-with-pytorch

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2206.02535v1)

**Authors**: Motasem Alfarra, Juan C. Pérez, Egor Shulgin, Peter Richtárik, Bernard Ghanem

**Abstracts**: Federated learning has recently gained significant attention and popularity due to its effectiveness in training machine learning models on distributed data privately. However, as in the single-node supervised learning setup, models trained in federated learning suffer from vulnerability to imperceptible input transformations known as adversarial attacks, questioning their deployment in security-related applications. In this work, we study the interplay between federated training, personalization, and certified robustness. In particular, we deploy randomized smoothing, a widely-used and scalable certification method, to certify deep networks trained on a federated setup against input perturbations and transformations. We find that the simple federated averaging technique is effective in building not only more accurate, but also more certifiably-robust models, compared to training solely on local data. We further analyze personalization, a popular technique in federated training that increases the model's bias towards local data, on robustness. We show several advantages of personalization over both~(that is, only training on local data and federated training) in building more robust models with faster training. Finally, we explore the robustness of mixtures of global and local~(\ie personalized) models, and find that the robustness of local models degrades as they diverge from the global model



## **24. Fast Adversarial Training with Adaptive Step Size**

cs.LG

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2206.02417v1)

**Authors**: Zhichao Huang, Yanbo Fan, Chen Liu, Weizhong Zhang, Yong Zhang, Mathieu Salzmann, Sabine Süsstrunk, Jue Wang

**Abstracts**: While adversarial training and its variants have shown to be the most effective algorithms to defend against adversarial attacks, their extremely slow training process makes it hard to scale to large datasets like ImageNet. The key idea of recent works to accelerate adversarial training is to substitute multi-step attacks (e.g., PGD) with single-step attacks (e.g., FGSM). However, these single-step methods suffer from catastrophic overfitting, where the accuracy against PGD attack suddenly drops to nearly 0% during training, destroying the robustness of the networks. In this work, we study the phenomenon from the perspective of training instances. We show that catastrophic overfitting is instance-dependent and fitting instances with larger gradient norm is more likely to cause catastrophic overfitting. Based on our findings, we propose a simple but effective method, Adversarial Training with Adaptive Step size (ATAS). ATAS learns an instancewise adaptive step size that is inversely proportional to its gradient norm. The theoretical analysis shows that ATAS converges faster than the commonly adopted non-adaptive counterparts. Empirically, ATAS consistently mitigates catastrophic overfitting and achieves higher robust accuracy on CIFAR10, CIFAR100 and ImageNet when evaluated on various adversarial budgets.



## **25. The art of defense: letting networks fool the attacker**

cs.CV

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2104.02963v3)

**Authors**: Jinlai Zhang, Yinpeng Dong, Binbin Liu, Bo Ouyang, Jihong Zhu, Minchi Kuang, Houqing Wang, Yanmei Meng

**Abstracts**: Robust environment perception is critical for autonomous cars, and adversarial defenses are the most effective and widely studied ways to improve the robustness of environment perception. However, all of previous defense methods decrease the natural accuracy, and the nature of the DNNs itself has been overlooked. To this end, in this paper, we propose a novel adversarial defense for 3D point cloud classifier that makes full use of the nature of the DNNs. Due to the disorder of point cloud, all point cloud classifiers have the property of permutation invariant to the input point cloud. Based on this nature, we design invariant transformations defense (IT-Defense). We show that, even after accounting for obfuscated gradients, our IT-Defense is a resilient defense against state-of-the-art (SOTA) 3D attacks. Moreover, IT-Defense do not hurt clean accuracy compared to previous SOTA 3D defenses. Our code is available at: {\footnotesize{\url{https://github.com/cuge1995/IT-Defense}}}.



## **26. Quantized and Distributed Subgradient Optimization Method with Malicious Attack**

math.OC

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2206.02272v1)

**Authors**: Iyanuoluwa Emiola, Chinwendu Enyioha

**Abstracts**: This paper considers a distributed optimization problem in a multi-agent system where a fraction of the agents act in an adversarial manner. Specifically, the malicious agents steer the network of agents away from the optimal solution by sending false information to their neighbors and consume significant bandwidth in the communication process. We propose a distributed gradient-based optimization algorithm in which the non-malicious agents exchange quantized information with one another. We prove convergence of the solution to a neighborhood of the optimal solution, and characterize the solutions obtained under the communication-constrained environment and presence of malicious agents. Numerical simulations to illustrate the results are also presented.



## **27. Vanilla Feature Distillation for Improving the Accuracy-Robustness Trade-Off in Adversarial Training**

cs.CV

12 pages

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2206.02158v1)

**Authors**: Guodong Cao, Zhibo Wang, Xiaowei Dong, Zhifei Zhang, Hengchang Guo, Zhan Qin, Kui Ren

**Abstracts**: Adversarial training has been widely explored for mitigating attacks against deep models. However, most existing works are still trapped in the dilemma between higher accuracy and stronger robustness since they tend to fit a model towards robust features (not easily tampered with by adversaries) while ignoring those non-robust but highly predictive features. To achieve a better robustness-accuracy trade-off, we propose the Vanilla Feature Distillation Adversarial Training (VFD-Adv), which conducts knowledge distillation from a pre-trained model (optimized towards high accuracy) to guide adversarial training towards higher accuracy, i.e., preserving those non-robust but predictive features. More specifically, both adversarial examples and their clean counterparts are forced to be aligned in the feature space by distilling predictive representations from the pre-trained/clean model, while previous works barely utilize predictive features from clean models. Therefore, the adversarial training model is updated towards maximally preserving the accuracy as gaining robustness. A key advantage of our method is that it can be universally adapted to and boost existing works. Exhaustive experiments on various datasets, classification models, and adversarial training algorithms demonstrate the effectiveness of our proposed method.



## **28. Federated Adversarial Training with Transformers**

cs.LG

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2206.02131v1)

**Authors**: Ahmed Aldahdooh, Wassim Hamidouche, Olivier Déforges

**Abstracts**: Federated learning (FL) has emerged to enable global model training over distributed clients' data while preserving its privacy. However, the global trained model is vulnerable to the evasion attacks especially, the adversarial examples (AEs), carefully crafted samples to yield false classification. Adversarial training (AT) is found to be the most promising approach against evasion attacks and it is widely studied for convolutional neural network (CNN). Recently, vision transformers have been found to be effective in many computer vision tasks. To the best of the authors' knowledge, there is no work that studied the feasibility of AT in a FL process for vision transformers. This paper investigates such feasibility with different federated model aggregation methods and different vision transformer models with different tokenization and classification head techniques. In order to improve the robust accuracy of the models with the not independent and identically distributed (Non-IID), we propose an extension to FedAvg aggregation method, called FedWAvg. By measuring the similarities between the last layer of the global model and the last layer of the client updates, FedWAvg calculates the weights to aggregate the local models updates. The experiments show that FedWAvg improves the robust accuracy when compared with other state-of-the-art aggregation methods.



## **29. Data-Efficient Backdoor Attacks**

cs.CV

Accepted to IJCAI 2022 Long Oral

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2204.12281v2)

**Authors**: Pengfei Xia, Ziqiang Li, Wei Zhang, Bin Li

**Abstracts**: Recent studies have proven that deep neural networks are vulnerable to backdoor attacks. Specifically, by mixing a small number of poisoned samples into the training set, the behavior of the trained model can be maliciously controlled. Existing attack methods construct such adversaries by randomly selecting some clean data from the benign set and then embedding a trigger into them. However, this selection strategy ignores the fact that each poisoned sample contributes inequally to the backdoor injection, which reduces the efficiency of poisoning. In this paper, we formulate improving the poisoned data efficiency by the selection as an optimization problem and propose a Filtering-and-Updating Strategy (FUS) to solve it. The experimental results on CIFAR-10 and ImageNet-10 indicate that the proposed method is effective: the same attack success rate can be achieved with only 47% to 75% of the poisoned sample volume compared to the random selection strategy. More importantly, the adversaries selected according to one setting can generalize well to other settings, exhibiting strong transferability. The prototype code of our method is now available at https://github.com/xpf/Data-Efficient-Backdoor-Attacks.



## **30. Connecting adversarial attacks and optimal transport for domain adaptation**

cs.LG

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2205.15424v2)

**Authors**: Arip Asadulaev, Vitaly Shutov, Alexander Korotin, Alexander Panfilov, Andrey Filchenkov

**Abstracts**: We present a novel algorithm for domain adaptation using optimal transport. In domain adaptation, the goal is to adapt a classifier trained on the source domain samples to the target domain. In our method, we use optimal transport to map target samples to the domain named source fiction. This domain differs from the source but is accurately classified by the source domain classifier. Our main idea is to generate a source fiction by c-cyclically monotone transformation over the target domain. If samples with the same labels in two domains are c-cyclically monotone, the optimal transport map between these domains preserves the class-wise structure, which is the main goal of domain adaptation. To generate a source fiction domain, we propose an algorithm that is based on our finding that adversarial attacks are a c-cyclically monotone transformation of the dataset. We conduct experiments on Digits and Modern Office-31 datasets and achieve improvement in performance for simple discrete optimal transport solvers for all adaptation tasks.



## **31. A General Framework for Evaluating Robustness of Combinatorial Optimization Solvers on Graphs**

math.OC

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2201.00402v2)

**Authors**: Han Lu, Zenan Li, Runzhong Wang, Qibing Ren, Junchi Yan, Xiaokang Yang

**Abstracts**: Solving combinatorial optimization (CO) on graphs is among the fundamental tasks for upper-stream applications in data mining, machine learning and operations research. Despite the inherent NP-hard challenge for CO, heuristics, branch-and-bound, learning-based solvers are developed to tackle CO problems as accurately as possible given limited time budgets. However, a practical metric for the sensitivity of CO solvers remains largely unexplored. Existing theoretical metrics require the optimal solution which is infeasible, and the gradient-based adversarial attack metric from deep learning is not compatible with non-learning solvers that are usually non-differentiable. In this paper, we develop the first practically feasible robustness metric for general combinatorial optimization solvers. We develop a no worse optimal cost guarantee thus do not require optimal solutions, and we tackle the non-differentiable challenge by resorting to black-box adversarial attack methods. Extensive experiments are conducted on 14 unique combinations of solvers and CO problems, and we demonstrate that the performance of state-of-the-art solvers like Gurobi can degenerate by over 20% under the given time limit bound on the hard instances discovered by our robustness metric, raising concerns about the robustness of combinatorial optimization solvers.



## **32. Guided Diffusion Model for Adversarial Purification**

cs.CV

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2205.14969v2)

**Authors**: Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, Hongfei Fu

**Abstracts**: With wider application of deep neural networks (DNNs) in various algorithms and frameworks, security threats have become one of the concerns. Adversarial attacks disturb DNN-based image classifiers, in which attackers can intentionally add imperceptible adversarial perturbations on input images to fool the classifiers. In this paper, we propose a novel purification approach, referred to as guided diffusion model for purification (GDMP), to help protect classifiers from adversarial attacks. The core of our approach is to embed purification into the diffusion denoising process of a Denoised Diffusion Probabilistic Model (DDPM), so that its diffusion process could submerge the adversarial perturbations with gradually added Gaussian noises, and both of these noises can be simultaneously removed following a guided denoising process. On our comprehensive experiments across various datasets, the proposed GDMP is shown to reduce the perturbations raised by adversarial attacks to a shallow range, thereby significantly improving the correctness of classification. GDMP improves the robust accuracy by 5%, obtaining 90.1% under PGD attack on the CIFAR10 dataset. Moreover, GDMP achieves 70.94% robustness on the challenging ImageNet dataset.



## **33. Soft Adversarial Training Can Retain Natural Accuracy**

cs.LG

7 pages, 6 figures

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2206.01904v1)

**Authors**: Abhijith Sharma, Apurva Narayan

**Abstracts**: Adversarial training for neural networks has been in the limelight in recent years. The advancement in neural network architectures over the last decade has led to significant improvement in their performance. It sparked an interest in their deployment for real-time applications. This process initiated the need to understand the vulnerability of these models to adversarial attacks. It is instrumental in designing models that are robust against adversaries. Recent works have proposed novel techniques to counter the adversaries, most often sacrificing natural accuracy. Most suggest training with an adversarial version of the inputs, constantly moving away from the original distribution. The focus of our work is to use abstract certification to extract a subset of inputs for (hence we call it 'soft') adversarial training. We propose a training framework that can retain natural accuracy without sacrificing robustness in a constrained setting. Our framework specifically targets moderately critical applications which require a reasonable balance between robustness and accuracy. The results testify to the idea of soft adversarial training for the defense against adversarial attacks. At last, we propose the scope of future work for further improvement of this framework.



## **34. Saliency Attack: Towards Imperceptible Black-box Adversarial Attack**

cs.LG

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2206.01898v1)

**Authors**: Zeyu Dai, Shengcai Liu, Ke Tang, Qing Li

**Abstracts**: Deep neural networks are vulnerable to adversarial examples, even in the black-box setting where the attacker is only accessible to the model output. Recent studies have devised effective black-box attacks with high query efficiency. However, such performance is often accompanied by compromises in attack imperceptibility, hindering the practical use of these approaches. In this paper, we propose to restrict the perturbations to a small salient region to generate adversarial examples that can hardly be perceived. This approach is readily compatible with many existing black-box attacks and can significantly improve their imperceptibility with little degradation in attack success rate. Further, we propose the Saliency Attack, a new black-box attack aiming to refine the perturbations in the salient region to achieve even better imperceptibility. Extensive experiments show that compared to the state-of-the-art black-box attacks, our approach achieves much better imperceptibility scores, including most apparent distortion (MAD), $L_0$ and $L_2$ distances, and also obtains significantly higher success rates judged by a human-like threshold on MAD. Importantly, the perturbations generated by our approach are interpretable to some extent. Finally, it is also demonstrated to be robust to different detection-based defenses.



## **35. Reward Poisoning Attacks on Offline Multi-Agent Reinforcement Learning**

cs.LG

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2206.01888v1)

**Authors**: Young Wu, Jermey McMahan, Xiaojin Zhu, Qiaomin Xie

**Abstracts**: We expose the danger of reward poisoning in offline multi-agent reinforcement learning (MARL), whereby an attacker can modify the reward vectors to different learners in an offline data set while incurring a poisoning cost. Based on the poisoned data set, all rational learners using some confidence-bound-based MARL algorithm will infer that a target policy - chosen by the attacker and not necessarily a solution concept originally - is the Markov perfect dominant strategy equilibrium for the underlying Markov Game, hence they will adopt this potentially damaging target policy in the future. We characterize the exact conditions under which the attacker can install a target policy. We further show how the attacker can formulate a linear program to minimize its poisoning cost. Our work shows the need for robust MARL against adversarial attacks.



## **36. Kallima: A Clean-label Framework for Textual Backdoor Attacks**

cs.CR

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01832v1)

**Authors**: Xiaoyi Chen, Yinpeng Dong, Zeyu Sun, Shengfang Zhai, Qingni Shen, Zhonghai Wu

**Abstracts**: Although Deep Neural Network (DNN) has led to unprecedented progress in various natural language processing (NLP) tasks, research shows that deep models are extremely vulnerable to backdoor attacks. The existing backdoor attacks mainly inject a small number of poisoned samples into the training dataset with the labels changed to the target one. Such mislabeled samples would raise suspicion upon human inspection, potentially revealing the attack. To improve the stealthiness of textual backdoor attacks, we propose the first clean-label framework Kallima for synthesizing mimesis-style backdoor samples to develop insidious textual backdoor attacks. We modify inputs belonging to the target class with adversarial perturbations, making the model rely more on the backdoor trigger. Our framework is compatible with most existing backdoor triggers. The experimental results on three benchmark datasets demonstrate the effectiveness of the proposed method.



## **37. Almost Tight L0-norm Certified Robustness of Top-k Predictions against Adversarial Perturbations**

cs.CR

Published as a conference paper at ICLR 2022

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2011.07633v2)

**Authors**: Jinyuan Jia, Binghui Wang, Xiaoyu Cao, Hongbin Liu, Neil Zhenqiang Gong

**Abstracts**: Top-k predictions are used in many real-world applications such as machine learning as a service, recommender systems, and web searches. $\ell_0$-norm adversarial perturbation characterizes an attack that arbitrarily modifies some features of an input such that a classifier makes an incorrect prediction for the perturbed input. $\ell_0$-norm adversarial perturbation is easy to interpret and can be implemented in the physical world. Therefore, certifying robustness of top-$k$ predictions against $\ell_0$-norm adversarial perturbation is important. However, existing studies either focused on certifying $\ell_0$-norm robustness of top-$1$ predictions or $\ell_2$-norm robustness of top-$k$ predictions. In this work, we aim to bridge the gap. Our approach is based on randomized smoothing, which builds a provably robust classifier from an arbitrary classifier via randomizing an input. Our major theoretical contribution is an almost tight $\ell_0$-norm certified robustness guarantee for top-$k$ predictions. We empirically evaluate our method on CIFAR10 and ImageNet. For instance, our method can build a classifier that achieves a certified top-3 accuracy of 69.2\% on ImageNet when an attacker can arbitrarily perturb 5 pixels of a testing image.



## **38. Gradient Obfuscation Checklist Test Gives a False Sense of Security**

cs.CV

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01705v1)

**Authors**: Nikola Popovic, Danda Pani Paudel, Thomas Probst, Luc Van Gool

**Abstracts**: One popular group of defense techniques against adversarial attacks is based on injecting stochastic noise into the network. The main source of robustness of such stochastic defenses however is often due to the obfuscation of the gradients, offering a false sense of security. Since most of the popular adversarial attacks are optimization-based, obfuscated gradients reduce their attacking ability, while the model is still susceptible to stronger or specifically tailored adversarial attacks. Recently, five characteristics have been identified, which are commonly observed when the improvement in robustness is mainly caused by gradient obfuscation. It has since become a trend to use these five characteristics as a sufficient test, to determine whether or not gradient obfuscation is the main source of robustness. However, these characteristics do not perfectly characterize all existing cases of gradient obfuscation, and therefore can not serve as a basis for a conclusive test. In this work, we present a counterexample, showing this test is not sufficient for concluding that gradient obfuscation is not the main cause of improvements in robustness.



## **39. Evaluating Transfer-based Targeted Adversarial Perturbations against Real-World Computer Vision Systems based on Human Judgments**

cs.CV

technical report

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01467v1)

**Authors**: Zhengyu Zhao, Nga Dang, Martha Larson

**Abstracts**: Computer vision systems are remarkably vulnerable to adversarial perturbations. Transfer-based adversarial images are generated on one (source) system and used to attack another (target) system. In this paper, we take the first step to investigate transfer-based targeted adversarial images in a realistic scenario where the target system is trained on some private data with its inventory of semantic labels not publicly available. Our main contributions include an extensive human-judgment-based evaluation of attack success on the Google Cloud Vision API and additional analysis of the different behaviors of Google Cloud Vision in face of original images vs. adversarial images. Resources are publicly available at \url{https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/google_results.zip}.



## **40. Adversarial Attacks on Human Vision**

cs.CV

21 pages, 8 figures, 1 table

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01365v1)

**Authors**: Victor A. Mateescu, Ivan V. Bajić

**Abstracts**: This article presents an introduction to visual attention retargeting, its connection to visual saliency, the challenges associated with it, and ideas for how it can be approached. The difficulty of attention retargeting as a saliency inversion problem lies in the lack of one-to-one mapping between saliency and the image domain, in addition to the possible negative impact of saliency alterations on image aesthetics. A few approaches from recent literature to solve this challenging problem are reviewed, and several suggestions for future development are presented.



## **41. On the Privacy Properties of GAN-generated Samples**

cs.LG

AISTATS 2021

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01349v1)

**Authors**: Zinan Lin, Vyas Sekar, Giulia Fanti

**Abstracts**: The privacy implications of generative adversarial networks (GANs) are a topic of great interest, leading to several recent algorithms for training GANs with privacy guarantees. By drawing connections to the generalization properties of GANs, we prove that under some assumptions, GAN-generated samples inherently satisfy some (weak) privacy guarantees. First, we show that if a GAN is trained on m samples and used to generate n samples, the generated samples are (epsilon, delta)-differentially-private for (epsilon, delta) pairs where delta scales as O(n/m). We show that under some special conditions, this upper bound is tight. Next, we study the robustness of GAN-generated samples to membership inference attacks. We model membership inference as a hypothesis test in which the adversary must determine whether a given sample was drawn from the training dataset or from the underlying data distribution. We show that this adversary can achieve an area under the ROC curve that scales no better than O(m^{-1/4}).



## **42. Adaptive Adversarial Training to Improve Adversarial Robustness of DNNs for Medical Image Segmentation and Detection**

eess.IV

8 pages

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.01736v1)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Recent methods based on Deep Neural Networks (DNNs) have reached high accuracy for medical image analysis, including the three basic tasks: segmentation, landmark detection, and object detection. It is known that DNNs are vulnerable to adversarial attacks, and the adversarial robustness of DNNs could be improved by adding adversarial noises to training data (i.e., adversarial training). In this study, we show that the standard adversarial training (SAT) method has a severe issue that limits its practical use: it generates a fixed level of noise for DNN training, and it is difficult for the user to choose an appropriate noise level, because a high noise level may lead to a large reduction in model performance, and a low noise level may have little effect. To resolve this issue, we have designed a novel adaptive-margin adversarial training (AMAT) method that generates adaptive adversarial noises for DNN training, which are dynamically tailored for each individual training sample. We have applied our AMAT method to state-of-the-art DNNs for the three basic tasks, using five publicly available datasets. The experimental results demonstrate that our AMAT method outperforms the SAT method in adversarial robustness on noisy data and prediction accuracy on clean data. Please contact the author for the source code.



## **43. A Barrier Certificate-based Simplex Architecture with Application to Microgrids**

eess.SY

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2202.09710v2)

**Authors**: Amol Damare, Shouvik Roy, Scott A. Smolka, Scott D. Stoller

**Abstracts**: We present Barrier Certificate-based Simplex (BC-Simplex), a new, provably correct design for runtime assurance of continuous dynamical systems. BC-Simplex is centered around the Simplex Control Architecture, which consists of a high-performance advanced controller which is not guaranteed to maintain safety of the plant, a verified-safe baseline controller, and a decision module that switches control of the plant between the two controllers to ensure safety without sacrificing performance. In BC-Simplex, Barrier certificates are used to prove that the baseline controller ensures safety. Furthermore, BC-Simplex features a new automated method for deriving, from the barrier certificate, the conditions for switching between the controllers. Our method is based on the Taylor expansion of the barrier certificate and yields computationally inexpensive switching conditions. We consider a significant application of BC-Simplex to a microgrid featuring an advanced controller in the form of a neural network trained using reinforcement learning. The microgrid is modeled in RTDS, an industry-standard high-fidelity, real-time power systems simulator. Our results demonstrate that BC-Simplex can automatically derive switching conditions for complex systems, the switching conditions are not overly conservative, and BC-Simplex ensures safety even in the presence of adversarial attacks on the neural controller.



## **44. Adversarial Laser Spot: Robust and Covert Physical Adversarial Attack to DNNs**

cs.CV

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.01034v1)

**Authors**: Chengyin Hu

**Abstracts**: Most existing deep neural networks (DNNs) are easily disturbed by slight noise. As far as we know, there are few researches on physical adversarial attack technology by deploying lighting equipment. The light-based physical adversarial attack technology has excellent covertness, which brings great security risks to many applications based on deep neural networks (such as automatic driving technology). Therefore, we propose a robust physical adversarial attack technology with excellent covertness, called adversarial laser point (AdvLS), which optimizes the physical parameters of laser point through genetic algorithm to perform physical adversarial attack. It realizes robust and covert physical adversarial attack by using low-cost laser equipment. As far as we know, AdvLS is the first light-based adversarial attack technology that can perform physical adversarial attacks in the daytime. A large number of experiments in the digital and physical environments show that AdvLS has excellent robustness and concealment. In addition, through in-depth analysis of the experimental data, we find that the adversarial perturbations generated by AdvLS have superior adversarial attack migration. The experimental results show that AdvLS impose serious interference to the advanced deep neural networks, we call for the attention of the proposed physical adversarial attack technology.



## **45. FACM: Correct the Output of Deep Neural Network with Middle Layers Features against Adversarial Samples**

cs.CV

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.00924v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: In the strong adversarial attacks against deep neural network (DNN), the output of DNN will be misclassified if and only if the last feature layer of the DNN is completely destroyed by adversarial samples, while our studies found that the middle feature layers of the DNN can still extract the effective features of the original normal category in these adversarial attacks. To this end, in this paper, a middle $\bold{F}$eature layer $\bold{A}$nalysis and $\bold{C}$onditional $\bold{M}$atching prediction distribution (FACM) model is proposed to increase the robustness of the DNN against adversarial samples through correcting the output of DNN with the features extracted by the middle layers of DNN. In particular, the middle $\bold{F}$eature layer $\bold{A}$nalysis (FA) module, the conditional matching prediction distribution (CMPD) module and the output decision module are included in our FACM model to collaboratively correct the classification of adversarial samples. The experiments results show that, our FACM model can significantly improve the robustness of the naturally trained model against various attacks, and our FA model can significantly improve the robustness of the adversarially trained model against white-box attacks with weak transferability and black box attacks where FA model includes the FA module and the output decision module, not the CMPD module.



## **46. Mask-Guided Divergence Loss Improves the Generalization and Robustness of Deep Neural Network**

cs.LG

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.00913v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Deep neural network (DNN) with dropout can be regarded as an ensemble model consisting of lots of sub-DNNs (i.e., an ensemble sub-DNN where the sub-DNN is the remaining part of the DNN after dropout), and through increasing the diversity of the ensemble sub-DNN, the generalization and robustness of the DNN can be effectively improved. In this paper, a mask-guided divergence loss function (MDL), which consists of a cross-entropy loss term and an orthogonal term, is proposed to increase the diversity of the ensemble sub-DNN by the added orthogonal term. Particularly, the mask technique is introduced to assist in generating the orthogonal term for avoiding overfitting of the diversity learning. The theoretical analysis and extensive experiments on 4 datasets (i.e., MNIST, FashionMNIST, CIFAR10, and CIFAR100) manifest that MDL can improve the generalization and robustness of standard training and adversarial training. For CIFAR10 and CIFAR100, in standard training, the maximum improvement of accuracy is $1.38\%$ on natural data, $30.97\%$ on FGSM (i.e., Fast Gradient Sign Method) attack, $38.18\%$ on PGD (i.e., Projected Gradient Descent) attack. While in adversarial training, the maximum improvement is $1.68\%$ on natural data, $4.03\%$ on FGSM attack and $2.65\%$ on PGD attack.



## **47. Adversarial RAW: Image-Scaling Attack Against Imaging Pipeline**

cs.CV

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.01733v1)

**Authors**: Junjian Li, Honglong Chen

**Abstracts**: Deep learning technologies have become the backbone for the development of computer vision. With further explorations, deep neural networks have been found vulnerable to well-designed adversarial attacks. Most of the vision devices are equipped with image signal processing (ISP) pipeline to implement RAW-to-RGB transformations and embedded into data preprocessing module for efficient image processing. Actually, ISP pipeline can introduce adversarial behaviors to post-capture images while data preprocessing may destroy attack patterns. However, none of the existing adversarial attacks takes into account the impacts of both ISP pipeline and data preprocessing. In this paper, we develop an image-scaling attack targeting on ISP pipeline, where the crafted adversarial RAW can be transformed into attack image that presents entirely different appearance once being scaled to a specific-size image. We first consider the gradient-available ISP pipeline, i.e., the gradient information can be directly used in the generation process of adversarial RAW to launch the attack. To make the adversarial attack more applicable, we further consider the gradient-unavailable ISP pipeline, in which a proxy model that well learns the RAW-to-RGB transformations is proposed as the gradient oracles. Extensive experiments show that the proposed adversarial attacks can craft adversarial RAW data against the target ISP pipelines with high attack rates.



## **48. Robust Feature-Level Adversaries are Interpretability Tools**

cs.LG

Code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2110.03605v4)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstracts**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying the representations in models. Second, we show that these adversaries are versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results indicate that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations.



## **49. On the reversibility of adversarial attacks**

cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00772v1)

**Authors**: Chau Yi Li, Ricardo Sánchez-Matilla, Ali Shahin Shamsabadi, Riccardo Mazzon, Andrea Cavallaro

**Abstracts**: Adversarial attacks modify images with perturbations that change the prediction of classifiers. These modified images, known as adversarial examples, expose the vulnerabilities of deep neural network classifiers. In this paper, we investigate the predictability of the mapping between the classes predicted for original images and for their corresponding adversarial examples. This predictability relates to the possibility of retrieving the original predictions and hence reversing the induced misclassification. We refer to this property as the reversibility of an adversarial attack, and quantify reversibility as the accuracy in retrieving the original class or the true class of an adversarial example. We present an approach that reverses the effect of an adversarial attack on a classifier using a prior set of classification results. We analyse the reversibility of state-of-the-art adversarial attacks on benchmark classifiers and discuss the factors that affect the reversibility.



## **50. Training privacy-preserving video analytics pipelines by suppressing features that reveal information about private attributes**

cs.CV

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2203.02635v2)

**Authors**: Chau Yi Li, Andrea Cavallaro

**Abstracts**: Deep neural networks are increasingly deployed for scene analytics, including to evaluate the attention and reaction of people exposed to out-of-home advertisements. However, the features extracted by a deep neural network that was trained to predict a specific, consensual attribute (e.g. emotion) may also encode and thus reveal information about private, protected attributes (e.g. age or gender). In this work, we focus on such leakage of private information at inference time. We consider an adversary with access to the features extracted by the layers of a deployed neural network and use these features to predict private attributes. To prevent the success of such an attack, we modify the training of the network using a confusion loss that encourages the extraction of features that make it difficult for the adversary to accurately predict private attributes. We validate this training approach on image-based tasks using a publicly available dataset. Results show that, compared to the original network, the proposed PrivateNet can reduce the leakage of private information of a state-of-the-art emotion recognition classifier by 2.88% for gender and by 13.06% for age group, with a minimal effect on task accuracy.



