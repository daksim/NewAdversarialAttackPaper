# Latest Adversarial Attack Papers
**update at 2023-05-20 15:39:52**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Attacks on Online Learners: a Teacher-Student Analysis**

stat.ML

15 pages, 6 figures

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11132v1) [paper-pdf](http://arxiv.org/pdf/2305.11132v1)

**Authors**: Riccardo Giuseppe Margiotta, Sebastian Goldt, Guido Sanguinetti

**Abstract**: Machine learning models are famously vulnerable to adversarial attacks: small ad-hoc perturbations of the data that can catastrophically alter the model predictions. While a large literature has studied the case of test-time attacks on pre-trained models, the important case of attacks in an online learning setting has received little attention so far. In this work, we use a control-theoretical perspective to study the scenario where an attacker may perturb data labels to manipulate the learning dynamics of an online learner. We perform a theoretical analysis of the problem in a teacher-student setup, considering different attack strategies, and obtaining analytical results for the steady state of simple linear learners. These results enable us to prove that a discontinuous transition in the learner's accuracy occurs when the attack strength exceeds a critical threshold. We then study empirically attacks on learners with complex architectures using real data, confirming the insights of our theoretical analysis. Our findings show that greedy attacks can be extremely efficient, especially when data stream in small batches.



## **2. Deep PackGen: A Deep Reinforcement Learning Framework for Adversarial Network Packet Generation**

cs.CR

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11039v1) [paper-pdf](http://arxiv.org/pdf/2305.11039v1)

**Authors**: Soumyadeep Hore, Jalal Ghadermazi, Diwas Paudel, Ankit Shah, Tapas K. Das, Nathaniel D. Bastian

**Abstract**: Recent advancements in artificial intelligence (AI) and machine learning (ML) algorithms, coupled with the availability of faster computing infrastructure, have enhanced the security posture of cybersecurity operations centers (defenders) through the development of ML-aided network intrusion detection systems (NIDS). Concurrently, the abilities of adversaries to evade security have also increased with the support of AI/ML models. Therefore, defenders need to proactively prepare for evasion attacks that exploit the detection mechanisms of NIDS. Recent studies have found that the perturbation of flow-based and packet-based features can deceive ML models, but these approaches have limitations. Perturbations made to the flow-based features are difficult to reverse-engineer, while samples generated with perturbations to the packet-based features are not playable.   Our methodological framework, Deep PackGen, employs deep reinforcement learning to generate adversarial packets and aims to overcome the limitations of approaches in the literature. By taking raw malicious network packets as inputs and systematically making perturbations on them, Deep PackGen camouflages them as benign packets while still maintaining their functionality. In our experiments, using publicly available data, Deep PackGen achieved an average adversarial success rate of 66.4\% against various ML models and across different attack types. Our investigation also revealed that more than 45\% of the successful adversarial samples were out-of-distribution packets that evaded the decision boundaries of the classifiers. The knowledge gained from our study on the adversary's ability to make specific evasive perturbations to different types of malicious packets can help defenders enhance the robustness of their NIDS against evolving adversarial attacks.



## **3. SoK: Data Privacy in Virtual Reality**

cs.HC

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2301.05940v2) [paper-pdf](http://arxiv.org/pdf/2301.05940v2)

**Authors**: Gonzalo Munilla Garrido, Vivek Nair, Dawn Song

**Abstract**: The adoption of virtual reality (VR) technologies has rapidly gained momentum in recent years as companies around the world begin to position the so-called "metaverse" as the next major medium for accessing and interacting with the internet. While consumers have become accustomed to a degree of data harvesting on the web, the real-time nature of data sharing in the metaverse indicates that privacy concerns are likely to be even more prevalent in the new "Web 3.0." Research into VR privacy has demonstrated that a plethora of sensitive personal information is observable by various would-be adversaries from just a few minutes of telemetry data. On the other hand, we have yet to see VR parallels for many privacy-preserving tools aimed at mitigating threats on conventional platforms. This paper aims to systematize knowledge on the landscape of VR privacy threats and countermeasures by proposing a comprehensive taxonomy of data attributes, protections, and adversaries based on the study of 68 collected publications. We complement our qualitative discussion with a statistical analysis of the risk associated with various data sources inherent to VR in consideration of the known attacks and defenses. By focusing on highlighting the clear outstanding opportunities, we hope to motivate and guide further research into this increasingly important field.



## **4. Certified Robust Neural Networks: Generalization and Corruption Resistance**

stat.ML

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2303.02251v2) [paper-pdf](http://arxiv.org/pdf/2303.02251v2)

**Authors**: Amine Bennouna, Ryan Lucas, Bart Van Parys

**Abstract**: Recent work have demonstrated that robustness (to "corruption") can be at odds with generalization. Adversarial training, for instance, aims to reduce the problematic susceptibility of modern neural networks to small data perturbations. Surprisingly, overfitting is a major concern in adversarial training despite being mostly absent in standard training. We provide here theoretical evidence for this peculiar "robust overfitting" phenomenon. Subsequently, we advance a novel distributionally robust loss function bridging robustness and generalization. We demonstrate both theoretically as well as empirically the loss to enjoy a certified level of robustness against two common types of corruption--data evasion and poisoning attacks--while ensuring guaranteed generalization. We show through careful numerical experiments that our resulting holistic robust (HR) training procedure yields SOTA performance. Finally, we indicate that HR training can be interpreted as a direct extension of adversarial training and comes with a negligible additional computational burden. A ready-to-use python library implementing our algorithm is available at https://github.com/RyanLucas3/HR_Neural_Networks.



## **5. Architecture-agnostic Iterative Black-box Certified Defense against Adversarial Patches**

cs.CV

9 pages

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10929v1) [paper-pdf](http://arxiv.org/pdf/2305.10929v1)

**Authors**: Di Yang, Yihao Huang, Qing Guo, Felix Juefei-Xu, Ming Hu, Yang Liu, Geguang Pu

**Abstract**: The adversarial patch attack aims to fool image classifiers within a bounded, contiguous region of arbitrary changes, posing a real threat to computer vision systems (e.g., autonomous driving, content moderation, biometric authentication, medical imaging) in the physical world. To address this problem in a trustworthy way, proposals have been made for certified patch defenses that ensure the robustness of classification models and prevent future patch attacks from breaching the defense. State-of-the-art certified defenses can be compatible with any model architecture, as well as achieve high clean and certified accuracy. Although the methods are adaptive to arbitrary patch positions, they inevitably need to access the size of the adversarial patch, which is unreasonable and impractical in real-world attack scenarios. To improve the feasibility of the architecture-agnostic certified defense in a black-box setting (i.e., position and size of the patch are both unknown), we propose a novel two-stage Iterative Black-box Certified Defense method, termed IBCD.In the first stage, it estimates the patch size in a search-based manner by evaluating the size relationship between the patch and mask with pixel masking. In the second stage, the accuracy results are calculated by the existing white-box certified defense methods with the estimated patch size. The experiments conducted on two popular model architectures and two datasets verify the effectiveness and efficiency of IBCD.



## **6. Free Lunch for Privacy Preserving Distributed Graph Learning**

cs.LG

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10869v1) [paper-pdf](http://arxiv.org/pdf/2305.10869v1)

**Authors**: Nimesh Agrawal, Nikita Malik, Sandeep Kumar

**Abstract**: Learning on graphs is becoming prevalent in a wide range of applications including social networks, robotics, communication, medicine, etc. These datasets belonging to entities often contain critical private information. The utilization of data for graph learning applications is hampered by the growing privacy concerns from users on data sharing. Existing privacy-preserving methods pre-process the data to extract user-side features, and only these features are used for subsequent learning. Unfortunately, these methods are vulnerable to adversarial attacks to infer private attributes. We present a novel privacy-respecting framework for distributed graph learning and graph-based machine learning. In order to perform graph learning and other downstream tasks on the server side, this framework aims to learn features as well as distances without requiring actual features while preserving the original structural properties of the raw data. The proposed framework is quite generic and highly adaptable. We demonstrate the utility of the Euclidean space, but it can be applied with any existing method of distance approximation and graph learning for the relevant spaces. Through extensive experimentation on both synthetic and real datasets, we demonstrate the efficacy of the framework in terms of comparing the results obtained without data sharing to those obtained with data sharing as a benchmark. This is, to our knowledge, the first privacy-preserving distributed graph learning framework.



## **7. How Deep Learning Sees the World: A Survey on Adversarial Attacks & Defenses**

cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10862v1) [paper-pdf](http://arxiv.org/pdf/2305.10862v1)

**Authors**: Joana C. Costa, Tiago Roxo, Hugo Proença, Pedro R. M. Inácio

**Abstract**: Deep Learning is currently used to perform multiple tasks, such as object recognition, face recognition, and natural language processing. However, Deep Neural Networks (DNNs) are vulnerable to perturbations that alter the network prediction (adversarial examples), raising concerns regarding its usage in critical areas, such as self-driving vehicles, malware detection, and healthcare. This paper compiles the most recent adversarial attacks, grouped by the attacker capacity, and modern defenses clustered by protection strategies. We also present the new advances regarding Vision Transformers, summarize the datasets and metrics used in the context of adversarial settings, and compare the state-of-the-art results under different attacks, finishing with the identification of open issues.



## **8. Towards an Accurate and Secure Detector against Adversarial Perturbations**

cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10856v1) [paper-pdf](http://arxiv.org/pdf/2305.10856v1)

**Authors**: Chao Wang, Shuren Qi, Zhiqiu Huang, Yushu Zhang, Xiaochun Cao

**Abstract**: The vulnerability of deep neural networks to adversarial perturbations has been widely perceived in the computer vision community. From a security perspective, it poses a critical risk for modern vision systems, e.g., the popular Deep Learning as a Service (DLaaS) frameworks. For protecting off-the-shelf deep models while not modifying them, current algorithms typically detect adversarial patterns through discriminative decomposition of natural-artificial data. However, these decompositions are biased towards frequency or spatial discriminability, thus failing to capture subtle adversarial patterns comprehensively. More seriously, they are typically invertible, meaning successful defense-aware (secondary) adversarial attack (i.e., evading the detector as well as fooling the model) is practical under the assumption that the adversary is fully aware of the detector (i.e., the Kerckhoffs's principle). Motivated by such facts, we propose an accurate and secure adversarial example detector, relying on a spatial-frequency discriminative decomposition with secret keys. It expands the above works on two aspects: 1) the introduced Krawtchouk basis provides better spatial-frequency discriminability and thereby is more suitable for capturing adversarial patterns than the common trigonometric or wavelet basis; 2) the extensive parameters for decomposition are generated by a pseudo-random function with secret keys, hence blocking the defense-aware adversarial attack. Theoretical and numerical analysis demonstrates the increased accuracy and security of our detector w.r.t. a number of state-of-the-art algorithms.



## **9. Adversarial Scratches: Deployable Attacks to CNN Classifiers**

cs.LG

This work is published at Pattern Recognition (Elsevier). This paper  stems from 'Scratch that! An Evolution-based Adversarial Attack against  Neural Networks' for which an arXiv preprint is available at  arXiv:1912.02316. Further studies led to a complete overhaul of the work,  resulting in this paper

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2204.09397v3) [paper-pdf](http://arxiv.org/pdf/2204.09397v3)

**Authors**: Loris Giulivi, Malhar Jere, Loris Rossi, Farinaz Koushanfar, Gabriela Ciocarlie, Briland Hitaj, Giacomo Boracchi

**Abstract**: A growing body of work has shown that deep neural networks are susceptible to adversarial examples. These take the form of small perturbations applied to the model's input which lead to incorrect predictions. Unfortunately, most literature focuses on visually imperceivable perturbations to be applied to digital images that often are, by design, impossible to be deployed to physical targets. We present Adversarial Scratches: a novel L0 black-box attack, which takes the form of scratches in images, and which possesses much greater deployability than other state-of-the-art attacks. Adversarial Scratches leverage B\'ezier Curves to reduce the dimension of the search space and possibly constrain the attack to a specific location. We test Adversarial Scratches in several scenarios, including a publicly available API and images of traffic signs. Results show that, often, our attack achieves higher fooling rate than other deployable state-of-the-art methods, while requiring significantly fewer queries and modifying very few pixels.



## **10. Adversarial Amendment is the Only Force Capable of Transforming an Enemy into a Friend**

cs.AI

Accepted to IJCAI 2023, 10 pages, 5 figures

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10766v1) [paper-pdf](http://arxiv.org/pdf/2305.10766v1)

**Authors**: Chong Yu, Tao Chen, Zhongxue Gan

**Abstract**: Adversarial attack is commonly regarded as a huge threat to neural networks because of misleading behavior. This paper presents an opposite perspective: adversarial attacks can be harnessed to improve neural models if amended correctly. Unlike traditional adversarial defense or adversarial training schemes that aim to improve the adversarial robustness, the proposed adversarial amendment (AdvAmd) method aims to improve the original accuracy level of neural models on benign samples. We thoroughly analyze the distribution mismatch between the benign and adversarial samples. This distribution mismatch and the mutual learning mechanism with the same learning ratio applied in prior art defense strategies is the main cause leading the accuracy degradation for benign samples. The proposed AdvAmd is demonstrated to steadily heal the accuracy degradation and even leads to a certain accuracy boost of common neural models on benign classification, object detection, and segmentation tasks. The efficacy of the AdvAmd is contributed by three key components: mediate samples (to reduce the influence of distribution mismatch with a fine-grained amendment), auxiliary batch norm (to solve the mutual learning mechanism and the smoother judgment surface), and AdvAmd loss (to adjust the learning ratios according to different attack vulnerabilities) through quantitative and ablation experiments.



## **11. Re-thinking Data Availablity Attacks Against Deep Neural Networks**

cs.CR

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10691v1) [paper-pdf](http://arxiv.org/pdf/2305.10691v1)

**Authors**: Bin Fang, Bo Li, Shuang Wu, Ran Yi, Shouhong Ding, Lizhuang Ma

**Abstract**: The unauthorized use of personal data for commercial purposes and the clandestine acquisition of private data for training machine learning models continue to raise concerns. In response to these issues, researchers have proposed availability attacks that aim to render data unexploitable. However, many current attack methods are rendered ineffective by adversarial training. In this paper, we re-examine the concept of unlearnable examples and discern that the existing robust error-minimizing noise presents an inaccurate optimization objective. Building on these observations, we introduce a novel optimization paradigm that yields improved protection results with reduced computational time requirements. We have conducted extensive experiments to substantiate the soundness of our approach. Moreover, our method establishes a robust foundation for future research in this area.



## **12. Content-based Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10665v1) [paper-pdf](http://arxiv.org/pdf/2305.10665v1)

**Authors**: Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4% and 16.8-48.0% in normally trained models and defense methods, respectively.



## **13. Exact Recovery for System Identification with More Corrupt Data than Clean Data**

cs.LG

24 pages, 2 figures

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2305.10506v1) [paper-pdf](http://arxiv.org/pdf/2305.10506v1)

**Authors**: Baturalp Yalcin, Javad Lavaei, Murat Arcak

**Abstract**: In this paper, we study the system identification problem for linear discrete-time systems under adversaries and analyze two lasso-type estimators. We study both asymptotic and non-asymptotic properties of these estimators in two separate scenarios, corresponding to deterministic and stochastic models for the attack times. Since the samples collected from the system are correlated, the existing results on lasso are not applicable. We show that when the system is stable and the attacks are injected periodically, the sample complexity for the exact recovery of the system dynamics is O(n), where n is the dimension of the states. When the adversarial attacks occur at each time instance with probability p, the required sample complexity for the exact recovery scales as O(\log(n)p/(1-p)^2). This result implies the almost sure convergence to the true system dynamics under the asymptotic regime. As a by-product, even when more than half of the data is compromised, our estimators still learn the system correctly. This paper provides the first mathematical guarantee in the literature on learning from correlated data for dynamical systems in the case when there is less clean data than corrupt data.



## **14. Raising the Bar for Certified Adversarial Robustness with Diffusion Models**

cs.LG

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2305.10388v1) [paper-pdf](http://arxiv.org/pdf/2305.10388v1)

**Authors**: Thomas Altstidl, David Dobre, Björn Eskofier, Gauthier Gidel, Leo Schwinn

**Abstract**: Certified defenses against adversarial attacks offer formal guarantees on the robustness of a model, making them more reliable than empirical methods such as adversarial training, whose effectiveness is often later reduced by unseen attacks. Still, the limited certified robustness that is currently achievable has been a bottleneck for their practical adoption. Gowal et al. and Wang et al. have shown that generating additional training data using state-of-the-art diffusion models can considerably improve the robustness of adversarial training. In this work, we demonstrate that a similar approach can substantially improve deterministic certified defenses. In addition, we provide a list of recommendations to scale the robustness of certified training approaches. One of our main insights is that the generalization gap, i.e., the difference between the training and test accuracy of the original model, is a good predictor of the magnitude of the robustness improvement when using additional generated data. Our approach achieves state-of-the-art deterministic robustness certificates on CIFAR-10 for the $\ell_2$ ($\epsilon = 36/255$) and $\ell_\infty$ ($\epsilon = 8/255$) threat models, outperforming the previous best results by $+3.95\%$ and $+1.39\%$, respectively. Furthermore, we report similar improvements for CIFAR-100.



## **15. Certified Invertibility in Neural Networks via Mixed-Integer Programming**

cs.LG

22 pages, 7 figures

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2301.11783v2) [paper-pdf](http://arxiv.org/pdf/2301.11783v2)

**Authors**: Tianqi Cui, Thomas Bertalan, George J. Pappas, Manfred Morari, Ioannis G. Kevrekidis, Mahyar Fazlyab

**Abstract**: Neural networks are known to be vulnerable to adversarial attacks, which are small, imperceptible perturbations that can significantly alter the network's output. Conversely, there may exist large, meaningful perturbations that do not affect the network's decision (excessive invariance). In our research, we investigate this latter phenomenon in two contexts: (a) discrete-time dynamical system identification, and (b) the calibration of a neural network's output to that of another network. We examine noninvertibility through the lens of mathematical optimization, where the global solution measures the ``safety" of the network predictions by their distance from the non-invertibility boundary. We formulate mixed-integer programs (MIPs) for ReLU networks and $L_p$ norms ($p=1,2,\infty$) that apply to neural network approximators of dynamical systems. We also discuss how our findings can be useful for invertibility certification in transformations between neural networks, e.g. between different levels of network pruning.



## **16. Manipulating Visually-aware Federated Recommender Systems and Its Countermeasures**

cs.IR

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.08183v2) [paper-pdf](http://arxiv.org/pdf/2305.08183v2)

**Authors**: Wei Yuan, Shilong Yuan, Chaoqun Yang, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Federated recommender systems (FedRecs) have been widely explored recently due to their ability to protect user data privacy. In FedRecs, a central server collaboratively learns recommendation models by sharing model public parameters with clients, thereby offering a privacy-preserving solution. Unfortunately, the exposure of model parameters leaves a backdoor for adversaries to manipulate FedRecs. Existing works about FedRec security already reveal that items can easily be promoted by malicious users via model poisoning attacks, but all of them mainly focus on FedRecs with only collaborative information (i.e., user-item interactions). We argue that these attacks are effective because of the data sparsity of collaborative signals. In practice, auxiliary information, such as products' visual descriptions, is used to alleviate collaborative filtering data's sparsity. Therefore, when incorporating visual information in FedRecs, all existing model poisoning attacks' effectiveness becomes questionable. In this paper, we conduct extensive experiments to verify that incorporating visual information can beat existing state-of-the-art attacks in reasonable settings. However, since visual information is usually provided by external sources, simply including it will create new security problems. Specifically, we propose a new kind of poisoning attack for visually-aware FedRecs, namely image poisoning attacks, where adversaries can gradually modify the uploaded image to manipulate item ranks during FedRecs' training process. Furthermore, we reveal that the potential collaboration between image poisoning attacks and model poisoning attacks will make visually-aware FedRecs more vulnerable to being manipulated. To safely use visual information, we employ a diffusion model in visually-aware FedRecs to purify each uploaded image and detect the adversarial images.



## **17. A theoretical basis for Blockchain Extractable Value**

cs.CR

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2302.02154v3) [paper-pdf](http://arxiv.org/pdf/2302.02154v3)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Extractable Value refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream protocols, like e.g. decentralized exchanges, are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the growing impact of these attacks in the real world, theoretical foundations are still missing. We propose a formal theory of Extractable Value, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against Extractable Value attacks.



## **18. Exploring the Connection between Robust and Generative Models**

cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2304.04033v3) [paper-pdf](http://arxiv.org/pdf/2304.04033v3)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.



## **19. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

cs.LG

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.09241v1) [paper-pdf](http://arxiv.org/pdf/2305.09241v1)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning.



## **20. Ortho-ODE: Enhancing Robustness and of Neural ODEs against Adversarial Attacks**

cs.LG

Final project paper

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.09179v1) [paper-pdf](http://arxiv.org/pdf/2305.09179v1)

**Authors**: Vishal Purohit

**Abstract**: Neural Ordinary Differential Equations (NODEs) probed the usage of numerical solvers to solve the differential equation characterized by a Neural Network (NN), therefore initiating a new paradigm of deep learning models with infinite depth. NODEs were designed to tackle the irregular time series problem. However, NODEs have demonstrated robustness against various noises and adversarial attacks. This paper is about the natural robustness of NODEs and examines the cause behind such surprising behaviour. We show that by controlling the Lipschitz constant of the ODE dynamics the robustness can be significantly improved. We derive our approach from Grownwall's inequality. Further, we draw parallels between contractivity theory and Grownwall's inequality. Experimentally we corroborate the enhanced robustness on numerous datasets - MNIST, CIFAR-10, and CIFAR 100. We also present the impact of adaptive and non-adaptive solvers on the robustness of NODEs.



## **21. Run-Off Election: Improved Provable Defense against Data Poisoning Attacks**

cs.LG

Accepted to ICML 2023

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2302.02300v3) [paper-pdf](http://arxiv.org/pdf/2302.02300v3)

**Authors**: Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini, Soheil Feizi

**Abstract**: In data poisoning attacks, an adversary tries to change a model's prediction by adding, modifying, or removing samples in the training data. Recently, ensemble-based approaches for obtaining provable defenses against data poisoning have been proposed where predictions are done by taking a majority vote across multiple base models. In this work, we show that merely considering the majority vote in ensemble defenses is wasteful as it does not effectively utilize available information in the logits layers of the base models. Instead, we propose Run-Off Election (ROE), a novel aggregation method based on a two-round election across the base models: In the first round, models vote for their preferred class and then a second, Run-Off election is held between the top two classes in the first round. Based on this approach, we propose DPA+ROE and FA+ROE defense methods based on Deep Partition Aggregation (DPA) and Finite Aggregation (FA) approaches from prior work. We evaluate our methods on MNIST, CIFAR-10, and GTSRB and obtain improvements in certified accuracy by up to 3%-4%. Also, by applying ROE on a boosted version of DPA, we gain improvements around 12%-27% comparing to the current state-of-the-art, establishing a new state-of-the-art in (pointwise) certified robustness against data poisoning. In many cases, our approach outperforms the state-of-the-art, even when using 32 times less computational power.



## **22. Training Neural Networks without Backpropagation: A Deeper Dive into the Likelihood Ratio Method**

cs.LG

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08960v1) [paper-pdf](http://arxiv.org/pdf/2305.08960v1)

**Authors**: Jinyang Jiang, Zeliang Zhang, Chenliang Xu, Zhaofei Yu, Yijie Peng

**Abstract**: Backpropagation (BP) is the most important gradient estimation method for training neural networks in deep learning. However, the literature shows that neural networks trained by BP are vulnerable to adversarial attacks. We develop the likelihood ratio (LR) method, a new gradient estimation method, for training a broad range of neural network architectures, including convolutional neural networks, recurrent neural networks, graph neural networks, and spiking neural networks, without recursive gradient computation. We propose three methods to efficiently reduce the variance of the gradient estimation in the neural network training process. Our experiments yield numerical results for training different neural networks on several datasets. All results demonstrate that the LR method is effective for training various neural networks and significantly improves the robustness of the neural networks under adversarial attacks relative to the BP method.



## **23. Attacking Perceptual Similarity Metrics**

cs.CV

TMLR 2023 (Featured Certification). Code is available at  https://tinyurl.com/attackingpsm

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08840v1) [paper-pdf](http://arxiv.org/pdf/2305.08840v1)

**Authors**: Abhijay Ghildyal, Feng Liu

**Abstract**: Perceptual similarity metrics have progressively become more correlated with human judgments on perceptual similarity; however, despite recent advances, the addition of an imperceptible distortion can still compromise these metrics. In our study, we systematically examine the robustness of these metrics to imperceptible adversarial perturbations. Following the two-alternative forced-choice experimental design with two distorted images and one reference image, we perturb the distorted image closer to the reference via an adversarial attack until the metric flips its judgment. We first show that all metrics in our study are susceptible to perturbations generated via common adversarial attacks such as FGSM, PGD, and the One-pixel attack. Next, we attack the widely adopted LPIPS metric using spatial-transformation-based adversarial perturbations (stAdv) in a white-box setting to craft adversarial examples that can effectively transfer to other similarity metrics in a black-box setting. We also combine the spatial attack stAdv with PGD ($\ell_\infty$-bounded) attack to increase transferability and use these adversarial examples to benchmark the robustness of both traditional and recently developed metrics. Our benchmark provides a good starting point for discussion and further research on the robustness of metrics to imperceptible adversarial perturbations.



## **24. Defending Against Misinformation Attacks in Open-Domain Question Answering**

cs.CL

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2212.10002v2) [paper-pdf](http://arxiv.org/pdf/2212.10002v2)

**Authors**: Orion Weller, Aleem Khan, Nathaniel Weir, Dawn Lawrie, Benjamin Van Durme

**Abstract**: Recent work in open-domain question answering (ODQA) has shown that adversarial poisoning of the search collection can cause large drops in accuracy for production systems. However, little to no work has proposed methods to defend against these attacks. To do so, we rely on the intuition that redundant information often exists in large corpora. To find it, we introduce a method that uses query augmentation to search for a diverse set of passages that could answer the original question but are less likely to have been poisoned. We integrate these new passages into the model through the design of a novel confidence method, comparing the predicted answer to its appearance in the retrieved contexts (what we call \textit{Confidence from Answer Redundancy}, i.e. CAR). Together these methods allow for a simple but effective way to defend against poisoning attacks that provides gains of nearly 20\% exact match across varying levels of data poisoning/knowledge conflicts.



## **25. Diffusion Models for Imperceptible and Transferable Adversarial Attack**

cs.CV

Code Page: https://github.com/WindVChen/DiffAttack

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08192v1) [paper-pdf](http://arxiv.org/pdf/2305.08192v1)

**Authors**: Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi

**Abstract**: Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models. Specifically, instead of direct manipulation in pixel space, we craft perturbations in latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an additional recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, DiffAttack, is the first that introduces diffusion models into adversarial attack field. Extensive experiments on various model structures (including CNNs, Transformers, MLPs) and defense methods have demonstrated our superiority over other attack methods.



## **26. Watermarking Text Generated by Black-Box Language Models**

cs.CL

Code will be available at  https://github.com/Kiode/Text_Watermark_Language_Models

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08883v1) [paper-pdf](http://arxiv.org/pdf/2305.08883v1)

**Authors**: Xi Yang, Kejiang Chen, Weiming Zhang, Chang Liu, Yuang Qi, Jie Zhang, Han Fang, Nenghai Yu

**Abstract**: LLMs now exhibit human-like skills in various fields, leading to worries about misuse. Thus, detecting generated text is crucial. However, passive detection methods are stuck in domain specificity and limited adversarial robustness. To achieve reliable detection, a watermark-based method was proposed for white-box LLMs, allowing them to embed watermarks during text generation. The method involves randomly dividing the model vocabulary to obtain a special list and adjusting the probability distribution to promote the selection of words in the list. A detection algorithm aware of the list can identify the watermarked text. However, this method is not applicable in many real-world scenarios where only black-box language models are available. For instance, third-parties that develop API-based vertical applications cannot watermark text themselves because API providers only supply generated text and withhold probability distributions to shield their commercial interests. To allow third-parties to autonomously inject watermarks into generated text, we develop a watermarking framework for black-box language model usage scenarios. Specifically, we first define a binary encoding function to compute a random binary encoding corresponding to a word. The encodings computed for non-watermarked text conform to a Bernoulli distribution, wherein the probability of a word representing bit-1 being approximately 0.5. To inject a watermark, we alter the distribution by selectively replacing words representing bit-0 with context-based synonyms that represent bit-1. A statistical test is then used to identify the watermark. Experiments demonstrate the effectiveness of our method on both Chinese and English datasets. Furthermore, results under re-translation, polishing, word deletion, and synonym substitution attacks reveal that it is arduous to remove the watermark without compromising the original semantics.



## **27. Improving Defensive Distillation using Teacher Assistant**

cs.CV

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08076v1) [paper-pdf](http://arxiv.org/pdf/2305.08076v1)

**Authors**: Maniratnam Mandal, Suna Gao

**Abstract**: Adversarial attacks pose a significant threat to the security and safety of deep neural networks being applied to modern applications. More specifically, in computer vision-based tasks, experts can use the knowledge of model architecture to create adversarial samples imperceptible to the human eye. These attacks can lead to security problems in popular applications such as self-driving cars, face recognition, etc. Hence, building networks which are robust to such attacks is highly desirable and essential. Among the various methods present in literature, defensive distillation has shown promise in recent years. Using knowledge distillation, researchers have been able to create models robust against some of those attacks. However, more attacks have been developed exposing weakness in defensive distillation. In this project, we derive inspiration from teacher assistant knowledge distillation and propose that introducing an assistant network can improve the robustness of the distilled model. Through a series of experiments, we evaluate the distilled models for different distillation temperatures in terms of accuracy, sensitivity, and robustness. Our experiments demonstrate that the proposed hypothesis can improve robustness in most cases. Additionally, we show that multi-step distillation can further improve robustness with very little impact on model accuracy.



## **28. DNN-Defender: An in-DRAM Deep Neural Network Defense Mechanism for Adversarial Weight Attack**

cs.CR

10 pages, 11 figures

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08034v1) [paper-pdf](http://arxiv.org/pdf/2305.08034v1)

**Authors**: Ranyang Zhou, Sabbir Ahmed, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: With deep learning deployed in many security-sensitive areas, machine learning security is becoming progressively important. Recent studies demonstrate attackers can exploit system-level techniques exploiting the RowHammer vulnerability of DRAM to deterministically and precisely flip bits in Deep Neural Networks (DNN) model weights to affect inference accuracy. The existing defense mechanisms are software-based, such as weight reconstruction requiring expensive training overhead or performance degradation. On the other hand, generic hardware-based victim-/aggressor-focused mechanisms impose expensive hardware overheads and preserve the spatial connection between victim and aggressor rows. In this paper, we present the first DRAM-based victim-focused defense mechanism tailored for quantized DNNs, named DNN-Defender that leverages the potential of in-DRAM swapping to withstand the targeted bit-flip attacks. Our results indicate that DNN-Defender can deliver a high level of protection downgrading the performance of targeted RowHammer attacks to a random attack level. In addition, the proposed defense has no accuracy drop on CIFAR-10 and ImageNet datasets without requiring any software training or incurring additional hardware overhead.



## **29. On enhancing the robustness of Vision Transformers: Defensive Diffusion**

cs.CV

Our code is publicly available at  https://github.com/Muhammad-Huzaifaa/Defensive_Diffusion

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08031v1) [paper-pdf](http://arxiv.org/pdf/2305.08031v1)

**Authors**: Raza Imam, Muhammad Huzaifa, Mohammed El-Amine Azz

**Abstract**: Privacy and confidentiality of medical data are of utmost importance in healthcare settings. ViTs, the SOTA vision model, rely on large amounts of patient data for training, which raises concerns about data security and the potential for unauthorized access. Adversaries may exploit vulnerabilities in ViTs to extract sensitive patient information and compromising patient privacy. This work address these vulnerabilities to ensure the trustworthiness and reliability of ViTs in medical applications. In this work, we introduced a defensive diffusion technique as an adversarial purifier to eliminate adversarial noise introduced by attackers in the original image. By utilizing the denoising capabilities of the diffusion model, we employ a reverse diffusion process to effectively eliminate the adversarial noise from the attack sample, resulting in a cleaner image that is then fed into the ViT blocks. Our findings demonstrate the effectiveness of the diffusion model in eliminating attack-agnostic adversarial noise from images. Additionally, we propose combining knowledge distillation with our framework to obtain a lightweight student model that is both computationally efficient and robust against gray box attacks. Comparison of our method with a SOTA baseline method, SEViT, shows that our work is able to outperform the baseline. Extensive experiments conducted on a publicly available Tuberculosis X-ray dataset validate the computational efficiency and improved robustness achieved by our proposed architecture.



## **30. Detection and Mitigation of Byzantine Attacks in Distributed Training**

cs.LG

21 pages, 17 figures, 6 tables. The material in this work appeared in  part at arXiv:2108.02416 which has been published at the 2022 IEEE  International Symposium on Information Theory

**SubmitDate**: 2023-05-13    [abs](http://arxiv.org/abs/2208.08085v4) [paper-pdf](http://arxiv.org/pdf/2208.08085v4)

**Authors**: Konstantinos Konstantinidis, Namrata Vaswani, Aditya Ramamoorthy

**Abstract**: A plethora of modern machine learning tasks require the utilization of large-scale distributed clusters as a critical component of the training pipeline. However, abnormal Byzantine behavior of the worker nodes can derail the training and compromise the quality of the inference. Such behavior can be attributed to unintentional system malfunctions or orchestrated attacks; as a result, some nodes may return arbitrary results to the parameter server (PS) that coordinates the training. Recent work considers a wide range of attack models and has explored robust aggregation and/or computational redundancy to correct the distorted gradients.   In this work, we consider attack models ranging from strong ones: $q$ omniscient adversaries with full knowledge of the defense protocol that can change from iteration to iteration to weak ones: $q$ randomly chosen adversaries with limited collusion abilities which only change every few iterations at a time. Our algorithms rely on redundant task assignments coupled with detection of adversarial behavior. We also show the convergence of our method to the optimal point under common assumptions and settings considered in literature. For strong attacks, we demonstrate a reduction in the fraction of distorted gradients ranging from 16%-99% as compared to the prior state-of-the-art. Our top-1 classification accuracy results on the CIFAR-10 data set demonstrate 25% advantage in accuracy (averaged over strong and weak scenarios) under the most sophisticated attacks compared to state-of-the-art methods.



## **31. Decision-based iterative fragile watermarking for model integrity verification**

cs.CR

**SubmitDate**: 2023-05-13    [abs](http://arxiv.org/abs/2305.09684v1) [paper-pdf](http://arxiv.org/pdf/2305.09684v1)

**Authors**: Zhaoxia Yin, Heng Yin, Hang Su, Xinpeng Zhang, Zhenzhe Gao

**Abstract**: Typically, foundation models are hosted on cloud servers to meet the high demand for their services. However, this exposes them to security risks, as attackers can modify them after uploading to the cloud or transferring from a local system. To address this issue, we propose an iterative decision-based fragile watermarking algorithm that transforms normal training samples into fragile samples that are sensitive to model changes. We then compare the output of sensitive samples from the original model to that of the compromised model during validation to assess the model's completeness.The proposed fragile watermarking algorithm is an optimization problem that aims to minimize the variance of the predicted probability distribution outputed by the target model when fed with the converted sample.We convert normal samples to fragile samples through multiple iterations. Our method has some advantages: (1) the iterative update of samples is done in a decision-based black-box manner, relying solely on the predicted probability distribution of the target model, which reduces the risk of exposure to adversarial attacks, (2) the small-amplitude multiple iterations approach allows the fragile samples to perform well visually, with a PSNR of 55 dB in TinyImageNet compared to the original samples, (3) even with changes in the overall parameters of the model of magnitude 1e-4, the fragile samples can detect such changes, and (4) the method is independent of the specific model structure and dataset. We demonstrate the effectiveness of our method on multiple models and datasets, and show that it outperforms the current state-of-the-art.



## **32. Quantum Lock: A Provable Quantum Communication Advantage**

quant-ph

47 pages, 13 figures

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2110.09469v4) [paper-pdf](http://arxiv.org/pdf/2110.09469v4)

**Authors**: Kaushik Chakraborty, Mina Doosti, Yao Ma, Chirag Wadhwa, Myrto Arapinis, Elham Kashefi

**Abstract**: Physical unclonable functions(PUFs) provide a unique fingerprint to a physical entity by exploiting the inherent physical randomness. Gao et al. discussed the vulnerability of most current-day PUFs to sophisticated machine learning-based attacks. We address this problem by integrating classical PUFs and existing quantum communication technology. Specifically, this paper proposes a generic design of provably secure PUFs, called hybrid locked PUFs(HLPUFs), providing a practical solution for securing classical PUFs. An HLPUF uses a classical PUF(CPUF), and encodes the output into non-orthogonal quantum states to hide the outcomes of the underlying CPUF from any adversary. Here we introduce a quantum lock to protect the HLPUFs from any general adversaries. The indistinguishability property of the non-orthogonal quantum states, together with the quantum lockdown technique prevents the adversary from accessing the outcome of the CPUFs. Moreover, we show that by exploiting non-classical properties of quantum states, the HLPUF allows the server to reuse the challenge-response pairs for further client authentication. This result provides an efficient solution for running PUF-based client authentication for an extended period while maintaining a small-sized challenge-response pairs database on the server side. Later, we support our theoretical contributions by instantiating the HLPUFs design using accessible real-world CPUFs. We use the optimal classical machine-learning attacks to forge both the CPUFs and HLPUFs, and we certify the security gap in our numerical simulation for construction which is ready for implementation.



## **33. Two-in-One: A Model Hijacking Attack Against Text Generation Models**

cs.CR

To appear in the 32nd USENIX Security Symposium, August 2023,  Anaheim, CA, USA

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07406v1) [paper-pdf](http://arxiv.org/pdf/2305.07406v1)

**Authors**: Wai Man Si, Michael Backes, Yang Zhang, Ahmed Salem

**Abstract**: Machine learning has progressed significantly in various applications ranging from face recognition to text generation. However, its success has been accompanied by different attacks. Recently a new attack has been proposed which raises both accountability and parasitic computing risks, namely the model hijacking attack. Nevertheless, this attack has only focused on image classification tasks. In this work, we broaden the scope of this attack to include text generation and classification models, hence showing its broader applicability. More concretely, we propose a new model hijacking attack, Ditto, that can hijack different text classification tasks into multiple generation ones, e.g., language translation, text summarization, and language modeling. We use a range of text benchmark datasets such as SST-2, TweetEval, AGnews, QNLI, and IMDB to evaluate the performance of our attacks. Our results show that by using Ditto, an adversary can successfully hijack text generation models without jeopardizing their utility.



## **34. Novel bribery mining attacks in the bitcoin system and the bribery miner's dilemma**

cs.GT

26 pages, 16 figures, 3 tables

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07381v1) [paper-pdf](http://arxiv.org/pdf/2305.07381v1)

**Authors**: Junjie Hu, Chunxiang Xu, Zhe Jiang, Jiwu Cao

**Abstract**: Mining attacks allow adversaries to obtain a disproportionate share of the mining reward by deviating from the honest mining strategy in the Bitcoin system. Among them, the most well-known are selfish mining (SM), block withholding (BWH), fork after withholding (FAW) and bribery mining. In this paper, we propose two novel mining attacks: bribery semi-selfish mining (BSSM) and bribery stubborn mining (BSM). Both of them can increase the relative extra reward of the adversary and will make the target bribery miners suffer from the bribery miner dilemma. All targets earn less under the Nash equilibrium. For each target, their local optimal strategy is to accept the bribes. However, they will suffer losses, comparing with denying the bribes. Furthermore, for all targets, their global optimal strategy is to deny the bribes. Quantitative analysis and simulation have been verified our theoretical analysis. We propose practical measures to mitigate more advanced mining attack strategies based on bribery mining, and provide new ideas for addressing bribery mining attacks in the future. However, how to completely and effectively prevent these attacks is still needed on further research.



## **35. Efficient Search of Comprehensively Robust Neural Architectures via Multi-fidelity Evaluation**

cs.CV

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07308v1) [paper-pdf](http://arxiv.org/pdf/2305.07308v1)

**Authors**: Jialiang Sun, Wen Yao, Tingsong Jiang, Xiaoqian Chen

**Abstract**: Neural architecture search (NAS) has emerged as one successful technique to find robust deep neural network (DNN) architectures. However, most existing robustness evaluations in NAS only consider $l_{\infty}$ norm-based adversarial noises. In order to improve the robustness of DNN models against multiple types of noises, it is necessary to consider a comprehensive evaluation in NAS for robust architectures. But with the increasing number of types of robustness evaluations, it also becomes more time-consuming to find comprehensively robust architectures. To alleviate this problem, we propose a novel efficient search of comprehensively robust neural architectures via multi-fidelity evaluation (ES-CRNA-ME). Specifically, we first search for comprehensively robust architectures under multiple types of evaluations using the weight-sharing-based NAS method, including different $l_{p}$ norm attacks, semantic adversarial attacks, and composite adversarial attacks. In addition, we reduce the number of robustness evaluations by the correlation analysis, which can incorporate similar evaluations and decrease the evaluation cost. Finally, we propose a multi-fidelity online surrogate during optimization to further decrease the search cost. On the basis of the surrogate constructed by low-fidelity data, the online high-fidelity data is utilized to finetune the surrogate. Experiments on CIFAR10 and CIFAR100 datasets show the effectiveness of our proposed method.



## **36. Parameter identifiability of a deep feedforward ReLU neural network**

math.ST

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2112.12982v2) [paper-pdf](http://arxiv.org/pdf/2112.12982v2)

**Authors**: Joachim Bona-Pellissier, François Bachoc, François Malgouyres

**Abstract**: The possibility for one to recover the parameters-weights and biases-of a neural network thanks to the knowledge of its function on a subset of the input space can be, depending on the situation, a curse or a blessing. On one hand, recovering the parameters allows for better adversarial attacks and could also disclose sensitive information from the dataset used to construct the network. On the other hand, if the parameters of a network can be recovered, it guarantees the user that the features in the latent spaces can be interpreted. It also provides foundations to obtain formal guarantees on the performances of the network. It is therefore important to characterize the networks whose parameters can be identified and those whose parameters cannot. In this article, we provide a set of conditions on a deep fully-connected feedforward ReLU neural network under which the parameters of the network are uniquely identified-modulo permutation and positive rescaling-from the function it implements on a subset of the input space.



## **37. Adversarial Security and Differential Privacy in mmWave Beam Prediction in 6G networks**

cs.CR

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.09679v1) [paper-pdf](http://arxiv.org/pdf/2305.09679v1)

**Authors**: Ghanta Sai Krishna, Kundrapu Supriya, Sanskar Singh, Sabur Baidya

**Abstract**: In the forthcoming era of 6G, the mmWave communication is envisioned to be used in dense user scenarios with high bandwidth requirements, that necessitate efficient and accurate beam prediction. Machine learning (ML) based approaches are ushering as a critical solution for achieving such efficient beam prediction for 6G mmWave communications. However, most contemporary ML classifiers are quite susceptible to adversarial inputs. Attackers can easily perturb the methodology through noise addition in the model itself. To mitigate this, the current work presents a defensive mechanism for attenuating the adversarial attacks against projected ML-based models for mmWave beam anticipation by incorporating adversarial training. Furthermore, as training 6G mmWave beam prediction model necessitates the use of large and comprehensive datasets that could include sensitive information regarding the user's location, differential privacy (DP) has been introduced as a technique to preserve the confidentiality of the information by purposefully adding a low sensitivity controlled noise in the datasets. It ensures that even if the information about a user location could be retrieved, the attacker would have no means to determine whether the information is significant or meaningless. With ray-tracing simulations for various outdoor and indoor scenarios, we illustrate the advantage of our proposed novel framework in terms of beam prediction accuracy and effective achievable rate while ensuring the security and privacy in communications.



## **38. Physical-layer Adversarial Robustness for Deep Learning-based Semantic Communications**

eess.SP

17 pages, 28 figures, accepted by IEEE jsac

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07220v1) [paper-pdf](http://arxiv.org/pdf/2305.07220v1)

**Authors**: Guoshun Nan, Zhichun Li, Jinli Zhai, Qimei Cui, Gong Chen, Xin Du, Xuefei Zhang, Xiaofeng Tao, Zhu Han, Tony Q. S. Quek

**Abstract**: End-to-end semantic communications (ESC) rely on deep neural networks (DNN) to boost communication efficiency by only transmitting the semantics of data, showing great potential for high-demand mobile applications. We argue that central to the success of ESC is the robust interpretation of conveyed semantics at the receiver side, especially for security-critical applications such as automatic driving and smart healthcare. However, robustifying semantic interpretation is challenging as ESC is extremely vulnerable to physical-layer adversarial attacks due to the openness of wireless channels and the fragileness of neural models. Toward ESC robustness in practice, we ask the following two questions: Q1: For attacks, is it possible to generate semantic-oriented physical-layer adversarial attacks that are imperceptible, input-agnostic and controllable? Q2: Can we develop a defense strategy against such semantic distortions and previously proposed adversaries? To this end, we first present MobileSC, a novel semantic communication framework that considers the computation and memory efficiency in wireless environments. Equipped with this framework, we propose SemAdv, a physical-layer adversarial perturbation generator that aims to craft semantic adversaries over the air with the abovementioned criteria, thus answering the Q1. To better characterize the realworld effects for robust training and evaluation, we further introduce a novel adversarial training method SemMixed to harden the ESC against SemAdv attacks and existing strong threats, thus answering the Q2. Extensive experiments on three public benchmarks verify the effectiveness of our proposed methods against various physical adversarial attacks. We also show some interesting findings, e.g., our MobileSC can even be more robust than classical block-wise communication systems in the low SNR regime.



## **39. Stratified Adversarial Robustness with Rejection**

cs.LG

Paper published at International Conference on Machine Learning  (ICML'23)

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.01139v2) [paper-pdf](http://arxiv.org/pdf/2305.01139v2)

**Authors**: Jiefeng Chen, Jayaram Raghuram, Jihye Choi, Xi Wu, Yingyu Liang, Somesh Jha

**Abstract**: Recently, there is an emerging interest in adversarially training a classifier with a rejection option (also known as a selective classifier) for boosting adversarial robustness. While rejection can incur a cost in many applications, existing studies typically associate zero cost with rejecting perturbed inputs, which can result in the rejection of numerous slightly-perturbed inputs that could be correctly classified. In this work, we study adversarially-robust classification with rejection in the stratified rejection setting, where the rejection cost is modeled by rejection loss functions monotonically non-increasing in the perturbation magnitude. We theoretically analyze the stratified rejection setting and propose a novel defense method -- Adversarial Training with Consistent Prediction-based Rejection (CPR) -- for building a robust selective classifier. Experiments on image datasets demonstrate that the proposed method significantly outperforms existing methods under strong adaptive attacks. For instance, on CIFAR-10, CPR reduces the total robust loss (for different rejection losses) by at least 7.3% under both seen and unseen attacks.



## **40. Improving Hyperspectral Adversarial Robustness Under Multiple Attacks**

cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2210.16346v4) [paper-pdf](http://arxiv.org/pdf/2210.16346v4)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation models classifying hyperspectral images (HSI) are vulnerable to adversarial examples. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease in performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network.



## **41. Untargeted Near-collision Attacks in Biometric Recognition**

cs.CR

Addition of results and correction of typos

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2304.01580v2) [paper-pdf](http://arxiv.org/pdf/2304.01580v2)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Kevin Thiry-Atighehchi

**Abstract**: A biometric recognition system can operate in two distinct modes, identification or verification. In the first mode, the system recognizes an individual by searching the enrolled templates of all the users for a match. In the second mode, the system validates a user's identity claim by comparing the fresh provided template with the enrolled template. The biometric transformation schemes usually produce binary templates that are better handled by cryptographic schemes, and the comparison is based on a distance that leaks information about the similarities between two biometric templates. Both the experimentally determined false match rate and false non-match rate through recognition threshold adjustment define the recognition accuracy, and hence the security of the system. To the best of our knowledge, few works provide a formal treatment of the security under minimum leakage of information, i.e., the binary outcome of a comparison with a threshold. In this paper, we rely on probabilistic modelling to quantify the security strength of binary templates. We investigate the influence of template size, database size and threshold on the probability of having a near-collision. We highlight several untargeted attacks on biometric systems considering naive and adaptive adversaries. Interestingly, these attacks can be launched both online and offline and, both in the identification mode and in the verification mode. We discuss the choice of parameters through the generic presented attacks.



## **42. Distracting Downpour: Adversarial Weather Attacks for Motion Estimation**

cs.CV

This work is a direct extension of our extended abstract from  arXiv:2210.11242

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06716v1) [paper-pdf](http://arxiv.org/pdf/2305.06716v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks on motion estimation, or optical flow, optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, adverse weather conditions constitute a much more realistic threat scenario. Hence, in this work, we present a novel attack on motion estimation that exploits adversarially optimized particles to mimic weather effects like snowflakes, rain streaks or fog clouds. At the core of our attack framework is a differentiable particle rendering system that integrates particles (i) consistently over multiple time steps (ii) into the 3D space (iii) with a photo-realistic appearance. Through optimization, we obtain adversarial weather that significantly impacts the motion estimation. Surprisingly, methods that previously showed good robustness towards small per-pixel perturbations are particularly vulnerable to adversarial weather. At the same time, augmenting the training with non-optimized weather increases a method's robustness towards weather effects and improves generalizability at almost no additional cost.



## **43. Beyond the Model: Data Pre-processing Attack to Deep Learning Models in Android Apps**

cs.CR

Accepted to AsiaCCS WorkShop on Secure and Trustworthy Deep Learning  Systems (SecTL 2023)

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.03963v2) [paper-pdf](http://arxiv.org/pdf/2305.03963v2)

**Authors**: Ye Sang, Yujin Huang, Shuo Huang, Helei Cui

**Abstract**: The increasing popularity of deep learning (DL) models and the advantages of computing, including low latency and bandwidth savings on smartphones, have led to the emergence of intelligent mobile applications, also known as DL apps, in recent years. However, this technological development has also given rise to several security concerns, including adversarial examples, model stealing, and data poisoning issues. Existing works on attacks and countermeasures for on-device DL models have primarily focused on the models themselves. However, scant attention has been paid to the impact of data processing disturbance on the model inference. This knowledge disparity highlights the need for additional research to fully comprehend and address security issues related to data processing for on-device models. In this paper, we introduce a data processing-based attacks against real-world DL apps. In particular, our attack could influence the performance and latency of the model without affecting the operation of a DL app. To demonstrate the effectiveness of our attack, we carry out an empirical study on 517 real-world DL apps collected from Google Play. Among 320 apps utilizing MLkit, we find that 81.56\% of them can be successfully attacked.   The results emphasize the importance of DL app developers being aware of and taking actions to secure on-device models from the perspective of data processing.



## **44. On the Robustness of Graph Neural Diffusion to Topology Perturbations**

cs.LG

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2209.07754v2) [paper-pdf](http://arxiv.org/pdf/2209.07754v2)

**Authors**: Yang Song, Qiyu Kang, Sijie Wang, Zhao Kai, Wee Peng Tay

**Abstract**: Neural diffusion on graphs is a novel class of graph neural networks that has attracted increasing attention recently. The capability of graph neural partial differential equations (PDEs) in addressing common hurdles of graph neural networks (GNNs), such as the problems of over-smoothing and bottlenecks, has been investigated but not their robustness to adversarial attacks. In this work, we explore the robustness properties of graph neural PDEs. We empirically demonstrate that graph neural PDEs are intrinsically more robust against topology perturbation as compared to other GNNs. We provide insights into this phenomenon by exploiting the stability of the heat semigroup under graph topology perturbations. We discuss various graph diffusion operators and relate them to existing graph neural PDEs. Furthermore, we propose a general graph neural PDE framework based on which a new class of robust GNNs can be defined. We verify that the new model achieves comparable state-of-the-art performance on several benchmark datasets.



## **45. Prevention of shoulder-surfing attacks using shifting condition using digraph substitution rules**

cs.CR

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06549v1) [paper-pdf](http://arxiv.org/pdf/2305.06549v1)

**Authors**: Amanul Islam, Fazidah Othman, Nazmus Sakib, Hafiz Md. Hasan Babu

**Abstract**: Graphical passwords are implemented as an alternative scheme to replace alphanumeric passwords to help users to memorize their password. However, most of the graphical password systems are vulnerable to shoulder-surfing attack due to the usage of the visual interface. In this research, a method that uses shifting condition with digraph substitution rules is proposed to address shoulder-surfing attack problem. The proposed algorithm uses both password images and decoy images throughout the user authentication procedure to confuse adversaries from obtaining the password images via direct observation or watching from a recorded session. The pass-images generated by this suggested algorithm are random and can only be generated if the algorithm is fully understood. As a result, adversaries will have no clue to obtain the right password images to log in. A user study was undertaken to assess the proposed method's effectiveness to avoid shoulder-surfing attacks. The results of the user study indicate that the proposed approach can withstand shoulder-surfing attacks (both direct observation and video recording method).The proposed method was tested and the results showed that it is able to resist shoulder-surfing and frequency of occurrence analysis attacks. Moreover, the experience gained in this research can be pervaded the gap on the realm of knowledge of the graphical password.



## **46. Inter-frame Accelerate Attack against Video Interpolation Models**

cs.CV

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06540v1) [paper-pdf](http://arxiv.org/pdf/2305.06540v1)

**Authors**: Junpei Liao, Zhikai Chen, Liang Yi, Wenyuan Yang, Baoyuan Wu, Xiaochun Cao

**Abstract**: Deep learning based video frame interpolation (VIF) method, aiming to synthesis the intermediate frames to enhance video quality, have been highly developed in the past few years. This paper investigates the adversarial robustness of VIF models. We apply adversarial attacks to VIF models and find that the VIF models are very vulnerable to adversarial examples. To improve attack efficiency, we suggest to make full use of the property of video frame interpolation task. The intuition is that the gap between adjacent frames would be small, leading to the corresponding adversarial perturbations being similar as well. Then we propose a novel attack method named Inter-frame Accelerate Attack (IAA) that initializes the perturbation as the perturbation for the previous adjacent frame and reduces the number of attack iterations. It is shown that our method can improve attack efficiency greatly while achieving comparable attack performance with traditional methods. Besides, we also extend our method to video recognition models which are higher level vision tasks and achieves great attack efficiency.



## **47. Improving Adversarial Robustness via Joint Classification and Multiple Explicit Detection Classes**

cs.CV

20 pages, 6 figures

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2210.14410v2) [paper-pdf](http://arxiv.org/pdf/2210.14410v2)

**Authors**: Sina Baharlouei, Fatemeh Sheikholeslami, Meisam Razaviyayn, Zico Kolter

**Abstract**: This work concerns the development of deep networks that are certifiably robust to adversarial attacks. Joint robust classification-detection was recently introduced as a certified defense mechanism, where adversarial examples are either correctly classified or assigned to the "abstain" class. In this work, we show that such a provable framework can benefit by extension to networks with multiple explicit abstain classes, where the adversarial examples are adaptively assigned to those. We show that naively adding multiple abstain classes can lead to "model degeneracy", then we propose a regularization approach and a training method to counter this degeneracy by promoting full use of the multiple abstain classes. Our experiments demonstrate that the proposed approach consistently achieves favorable standard vs. robust verified accuracy tradeoffs, outperforming state-of-the-art algorithms for various choices of number of abstain classes.



## **48. Towards Adversarial-Resilient Deep Neural Networks for False Data Injection Attack Detection in Power Grids**

cs.CR

This paper has been accepted by the the 32nd International Conference  on Computer Communications and Networks (ICCCN 2023)

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2102.09057v2) [paper-pdf](http://arxiv.org/pdf/2102.09057v2)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Stella Sun, Kevin Tomsovic, Hairong Qi

**Abstract**: False data injection attacks (FDIAs) pose a significant security threat to power system state estimation. To detect such attacks, recent studies have proposed machine learning (ML) techniques, particularly deep neural networks (DNNs). However, most of these methods fail to account for the risk posed by adversarial measurements, which can compromise the reliability of DNNs in various ML applications. In this paper, we present a DNN-based FDIA detection approach that is resilient to adversarial attacks. We first analyze several adversarial defense mechanisms used in computer vision and show their inherent limitations in FDIA detection. We then propose an adversarial-resilient DNN detection framework for FDIA that incorporates random input padding in both the training and inference phases. Our simulations, based on an IEEE standard power system, demonstrate that this framework significantly reduces the effectiveness of adversarial attacks while having a negligible impact on the DNNs' detection performance.



## **49. Invisible Backdoor Attack with Dynamic Triggers against Person Re-identification**

cs.CV

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2211.10933v2) [paper-pdf](http://arxiv.org/pdf/2211.10933v2)

**Authors**: Wenli Sun, Xinyang Jiang, Shuguang Dou, Dongsheng Li, Duoqian Miao, Cheng Deng, Cairong Zhao

**Abstract**: In recent years, person Re-identification (ReID) has rapidly progressed with wide real-world applications, but also poses significant risks of adversarial attacks. In this paper, we focus on the backdoor attack on deep ReID models. Existing backdoor attack methods follow an all-to-one or all-to-all attack scenario, where all the target classes in the test set have already been seen in the training set. However, ReID is a much more complex fine-grained open-set recognition problem, where the identities in the test set are not contained in the training set. Thus, previous backdoor attack methods for classification are not applicable for ReID. To ameliorate this issue, we propose a novel backdoor attack on deep ReID under a new all-to-unknown scenario, called Dynamic Triggers Invisible Backdoor Attack (DT-IBA). Instead of learning fixed triggers for the target classes from the training set, DT-IBA can dynamically generate new triggers for any unknown identities. Specifically, an identity hashing network is proposed to first extract target identity information from a reference image, which is then injected into the benign images by image steganography. We extensively validate the effectiveness and stealthiness of the proposed attack on benchmark datasets, and evaluate the effectiveness of several defense methods against our attack.



## **50. The Robustness of Computer Vision Models against Common Corruptions: a Survey**

cs.CV

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.06024v1) [paper-pdf](http://arxiv.org/pdf/2305.06024v1)

**Authors**: Shunxin Wang, Raymond Veldhuis, Nicola Strisciuglio

**Abstract**: The performance of computer vision models is susceptible to unexpected changes in input images when deployed in real scenarios. These changes are referred to as common corruptions. While they can hinder the applicability of computer vision models in real-world scenarios, they are not always considered as a testbed for model generalization and robustness. In this survey, we present a comprehensive and systematic overview of methods that improve corruption robustness of computer vision models. Unlike existing surveys that focus on adversarial attacks and label noise, we cover extensively the study of robustness to common corruptions that can occur when deploying computer vision models to work in practical applications. We describe different types of image corruption and provide the definition of corruption robustness. We then introduce relevant evaluation metrics and benchmark datasets. We categorize methods into four groups. We also cover indirect methods that show improvements in generalization and may improve corruption robustness as a byproduct. We report benchmark results collected from the literature and find that they are not evaluated in a unified manner, making it difficult to compare and analyze. We thus built a unified benchmark framework to obtain directly comparable results on benchmark datasets. Furthermore, we evaluate relevant backbone networks pre-trained on ImageNet using our framework, providing an overview of the base corruption robustness of existing models to help choose appropriate backbones for computer vision tasks. We identify that developing methods to handle a wide range of corruptions and efficiently learn with limited data and computational resources is crucial for future development. Additionally, we highlight the need for further investigation into the relationship among corruption robustness, OOD generalization, and shortcut learning.



