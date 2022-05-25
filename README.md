# Latest Adversarial Attack Papers
**update at 2022-05-26 06:31:30**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Self-Supervised Contrastive Learning with Adversarial Perturbations for Defending Word Substitution-based Attacks**

cs.CL

In Findings of NAACL 2022

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2107.07610v3)

**Authors**: Zhao Meng, Yihan Dong, Mrinmaya Sachan, Roger Wattenhofer

**Abstracts**: In this paper, we present an approach to improve the robustness of BERT language models against word substitution-based adversarial attacks by leveraging adversarial perturbations for self-supervised contrastive learning. We create a word-level adversarial attack generating hard positives on-the-fly as adversarial examples during contrastive learning. In contrast to previous works, our method improves model robustness without using any labeled data. Experimental results show that our method improves robustness of BERT against four different word substitution-based adversarial attacks, and combining our method with adversarial training gives higher robustness than adversarial training alone. As our method improves the robustness of BERT purely with unlabeled data, it opens up the possibility of using large text datasets to train robust language models against word substitution-based adversarial attacks.



## **2. Adversarial Attack on Attackers: Post-Process to Mitigate Black-Box Score-Based Query Attacks**

cs.LG

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.12134v1)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Yingwen Wu, Cihang Xie, Xiaolin Huang

**Abstracts**: The score-based query attacks (SQAs) pose practical threats to deep neural networks by crafting adversarial perturbations within dozens of queries, only using the model's output scores. Nonetheless, we note that if the loss trend of the outputs is slightly perturbed, SQAs could be easily misled and thereby become much less effective. Following this idea, we propose a novel defense, namely Adversarial Attack on Attackers (AAA), to confound SQAs towards incorrect attack directions by slightly modifying the output logits. In this way, (1) SQAs are prevented regardless of the model's worst-case robustness; (2) the original model predictions are hardly changed, i.e., no degradation on clean accuracy; (3) the calibration of confidence scores can be improved simultaneously. Extensive experiments are provided to verify the above advantages. For example, by setting $\ell_\infty=8/255$ on CIFAR-10, our proposed AAA helps WideResNet-28 secure $80.59\%$ accuracy under Square attack ($2500$ queries), while the best prior defense (i.e., adversarial training) only attains $67.44\%$. Since AAA attacks SQA's general greedy strategy, such advantages of AAA over 8 defenses can be consistently observed on 8 CIFAR-10/ImageNet models under 6 SQAs, using different attack targets and bounds. Moreover, AAA calibrates better without hurting the accuracy. Our code would be released.



## **3. Defending a Music Recommender Against Hubness-Based Adversarial Attacks**

eess.AS

6 pages, to be published in Proceedings of the 19th Sound and Music  Computing Conference 2022 (SMC-22)

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.12032v1)

**Authors**: Katharina Hoedt, Arthur Flexer, Gerhard Widmer

**Abstracts**: Adversarial attacks can drastically degrade performance of recommenders and other machine learning systems, resulting in an increased demand for defence mechanisms. We present a new line of defence against attacks which exploit a vulnerability of recommenders that operate in high dimensional data spaces (the so-called hubness problem). We use a global data scaling method, namely Mutual Proximity (MP), to defend a real-world music recommender which previously was susceptible to attacks that inflated the number of times a particular song was recommended. We find that using MP as a defence greatly increases robustness of the recommender against a range of attacks, with success rates of attacks around 44% (before defence) dropping to less than 6% (after defence). Additionally, adversarial examples still able to fool the defended system do so at the price of noticeably lower audio quality as shown by a decreased average SNR.



## **4. Phrase-level Textual Adversarial Attack with Label Preservation**

cs.CL

NAACL-HLT 2022 Findings (Long), 9 pages + 2 pages references + 8  pages appendix

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.10710v2)

**Authors**: Yibin Lei, Yu Cao, Dianqi Li, Tianyi Zhou, Meng Fang, Mykola Pechenizkiy

**Abstracts**: Generating high-quality textual adversarial examples is critical for investigating the pitfalls of natural language processing (NLP) models and further promoting their robustness. Existing attacks are usually realized through word-level or sentence-level perturbations, which either limit the perturbation space or sacrifice fluency and textual quality, both affecting the attack effectiveness. In this paper, we propose Phrase-Level Textual Adversarial aTtack (PLAT) that generates adversarial samples through phrase-level perturbations. PLAT first extracts the vulnerable phrases as attack targets by a syntactic parser, and then perturbs them by a pre-trained blank-infilling model. Such flexible perturbation design substantially expands the search space for more effective attacks without introducing too many modifications, and meanwhile maintaining the textual fluency and grammaticality via contextualized generation using surrounding texts. Moreover, we develop a label-preservation filter leveraging the likelihoods of language models fine-tuned on each class, rather than textual similarity, to rule out those perturbations that potentially alter the original class label for humans. Extensive experiments and human evaluation demonstrate that PLAT has a superior attack effectiveness as well as a better label consistency than strong baselines.



## **5. Can Adversarial Training Be Manipulated By Non-Robust Features?**

cs.LG

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2201.13329v2)

**Authors**: Lue Tao, Lei Feng, Hongxin Wei, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstracts**: Adversarial training, originally designed to resist test-time adversarial examples, has shown to be promising in mitigating training-time availability attacks. This defense ability, however, is challenged in this paper. We identify a novel threat model named stability attacks, which aims to hinder robust availability by slightly manipulating the training data. Under this threat, we show that adversarial training using a conventional defense budget $\epsilon$ provably fails to provide test robustness in a simple statistical setting, where the non-robust features of the training data can be reinforced by $\epsilon$-bounded perturbation. Further, we analyze the necessity of enlarging the defense budget to counter stability attacks. Finally, comprehensive experiments demonstrate that stability attacks are harmful on benchmark datasets, and thus the adaptive defense is necessary to maintain robustness.



## **6. Smart Grid: Cyber Attacks, Critical Defense Approaches, and Digital Twin**

cs.CR

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.11783v1)

**Authors**: Tianming Zheng, Ming Liu, Deepak Puthal, Ping Yi, Yue Wu, Xiangjian He

**Abstracts**: As a national critical infrastructure, the smart grid has attracted widespread attention for its cybersecurity issues. The development towards an intelligent, digital, and Internetconnected smart grid has attracted external adversaries for malicious activities. It is necessary to enhance its cybersecurity by either improving the existing defense approaches or introducing novel developed technologies to the smart grid context. As an emerging technology, digital twin (DT) is considered as an enabler for enhanced security. However, the practical implementation is quite challenging. This is due to the knowledge barriers among smart grid designers, security experts, and DT developers. Each single domain is a complicated system covering various components and technologies. As a result, works are needed to sort out relevant contents so that DT can be better embedded in the security architecture design of smart grid. In order to meet this demand, our paper covers the above three domains, i.e., smart grid, cybersecurity, and DT. Specifically, the paper i) introduces the background of the smart grid; ii) reviews external cyber attacks from attack incidents and attack methods; iii) introduces critical defense approaches in industrial cyber systems, which include device identification, vulnerability discovery, intrusion detection systems (IDSs), honeypots, attribution, and threat intelligence (TI); iv) reviews the relevant content of DT, including its basic concepts, applications in the smart grid, and how DT enhances the security. In the end, the paper puts forward our security considerations on the future development of DT-based smart grid. The survey is expected to help developers break knowledge barriers among smart grid, cybersecurity, and DT, and provide guidelines for future security design of DT-based smart grid.



## **7. Alleviating Robust Overfitting of Adversarial Training With Consistency Regularization**

cs.LG

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.11744v1)

**Authors**: Shudong Zhang, Haichang Gao, Tianwei Zhang, Yunyi Zhou, Zihui Wu

**Abstracts**: Adversarial training (AT) has proven to be one of the most effective ways to defend Deep Neural Networks (DNNs) against adversarial attacks. However, the phenomenon of robust overfitting, i.e., the robustness will drop sharply at a certain stage, always exists during AT. It is of great importance to decrease this robust generalization gap in order to obtain a robust model. In this paper, we present an in-depth study towards the robust overfitting from a new angle. We observe that consistency regularization, a popular technique in semi-supervised learning, has a similar goal as AT and can be used to alleviate robust overfitting. We empirically validate this observation, and find a majority of prior solutions have implicit connections to consistency regularization. Motivated by this, we introduce a new AT solution, which integrates the consistency regularization and Mean Teacher (MT) strategy into AT. Specifically, we introduce a teacher model, coming from the average weights of the student models over the training steps. Then we design a consistency loss function to make the prediction distribution of the student models over adversarial examples consistent with that of the teacher model over clean samples. Experiments show that our proposed method can effectively alleviate robust overfitting and improve the robustness of DNN models against common adversarial attacks.



## **8. Learning to Ignore Adversarial Attacks**

cs.CL

14 pages, 2 figures

**SubmitDate**: 2022-05-23    [paper-pdf](http://arxiv.org/pdf/2205.11551v1)

**Authors**: Yiming Zhang, Yangqiaoyu Zhou, Samuel Carton, Chenhao Tan

**Abstracts**: Despite the strong performance of current NLP models, they can be brittle against adversarial attacks. To enable effective learning against adversarial inputs, we introduce the use of rationale models that can explicitly learn to ignore attack tokens. We find that the rationale models can successfully ignore over 90\% of attack tokens. This approach leads to consistent sizable improvements ($\sim$10\%) over baseline models in robustness on three datasets for both BERT and RoBERTa, and also reliably outperforms data augmentation with adversarial examples alone. In many cases, we find that our method is able to close the gap between model performance on a clean test set and an attacked test set and hence reduce the effect of adversarial attacks.



## **9. Graph Layer Security: Encrypting Information via Common Networked Physics**

eess.SP

**SubmitDate**: 2022-05-23    [paper-pdf](http://arxiv.org/pdf/2006.03568v3)

**Authors**: Zhuangkun Wei, Liang Wang, Schyler Chengyao Sun, Bin Li, Weisi Guo

**Abstracts**: The proliferation of low-cost Internet of Things (IoT) devices has led to a race between wireless security and channel attacks. Traditional cryptography requires high-computational power and is not suitable for low-power IoT scenarios. Whist, recently developed physical layer security (PLS) can exploit common wireless channel state information (CSI), its sensitivity to channel estimation makes them vulnerable from attacks. In this work, we exploit an alternative common physics shared between IoT transceivers: the monitored channel-irrelevant physical networked dynamics (e.g., water/oil/gas/electrical signal-flows). Leveraging this, we propose for the first time, graph layer security (GLS), by exploiting the dependency in physical dynamics among network nodes for information encryption and decryption. A graph Fourier transform (GFT) operator is used to characterize such dependency into a graph-bandlimted subspace, which allows the generations of channel-irrelevant cipher keys by maximizing the secrecy rate. We evaluate our GLS against designed active and passive attackers, using IEEE 39-Bus system. Results demonstrate that, GLS is not reliant on wireless CSI, and can combat attackers that have partial networked dynamic knowledge (realistic access to full dynamic and critical nodes remains challenging). We believe this novel GLS has widespread applicability in secure health monitoring and for Digital Twins in adversarial radio environments.



## **10. Detection of Stealthy Adversaries for Networked Unmanned Aerial Vehicles***

eess.SY

to appear at the 2022 Int'l Conference on Unmanned Aircraft Systems  (ICUAS)

**SubmitDate**: 2022-05-22    [paper-pdf](http://arxiv.org/pdf/2202.09661v2)

**Authors**: Mohammad Bahrami, Hamidreza Jafarnejadsani

**Abstracts**: A network of unmanned aerial vehicles (UAVs) provides distributed coverage, reconfigurability, and maneuverability in performing complex cooperative tasks. However, it relies on wireless communications that can be susceptible to cyber adversaries and intrusions, disrupting the entire network's operation. This paper develops model-based centralized and decentralized observer techniques for detecting a class of stealthy intrusions, namely zero-dynamics and covert attacks, on networked UAVs in formation control settings. The centralized observer that runs in a control center leverages switching in the UAVs' communication topology for attack detection, and the decentralized observers, implemented onboard each UAV in the network, use the model of networked UAVs and locally available measurements. Experimental results are provided to show the effectiveness of the proposed detection schemes in different case studies.



## **11. Inverse-Inverse Reinforcement Learning. How to Hide Strategy from an Adversarial Inverse Reinforcement Learner**

cs.LG

**SubmitDate**: 2022-05-22    [paper-pdf](http://arxiv.org/pdf/2205.10802v1)

**Authors**: Kunal Pattanayak, Vikram Krishnamurthy, Christopher Berry

**Abstracts**: Inverse reinforcement learning (IRL) deals with estimating an agent's utility function from its actions. In this paper, we consider how an agent can hide its strategy and mitigate an adversarial IRL attack; we call this inverse IRL (I-IRL). How should the decision maker choose its response to ensure a poor reconstruction of its strategy by an adversary performing IRL to estimate the agent's strategy? This paper comprises four results: First, we present an adversarial IRL algorithm that estimates the agent's strategy while controlling the agent's utility function. Our second result for I-IRL result spoofs the IRL algorithm used by the adversary. Our I-IRL results are based on revealed preference theory in micro-economics. The key idea is for the agent to deliberately choose sub-optimal responses that sufficiently masks its true strategy. Third, we give a sample complexity result for our main I-IRL result when the agent has noisy estimates of the adversary specified utility function. Finally, we illustrate our I-IRL scheme in a radar problem where a meta-cognitive radar is trying to mitigate an adversarial target.



## **12. Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models**

cs.CR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10686v1)

**Authors**: Shawn Shan, Wenxin Ding, Emily Wenger, Haitao Zheng, Ben Y. Zhao

**Abstracts**: Server breaches are an unfortunate reality on today's Internet. In the context of deep neural network (DNN) models, they are particularly harmful, because a leaked model gives an attacker "white-box" access to generate adversarial examples, a threat model that has no practical robust defenses. For practitioners who have invested years and millions into proprietary DNNs, e.g. medical imaging, this seems like an inevitable disaster looming on the horizon.   In this paper, we consider the problem of post-breach recovery for DNN models. We propose Neo, a new system that creates new versions of leaked models, alongside an inference time filter that detects and removes adversarial examples generated on previously leaked models. The classification surfaces of different model versions are slightly offset (by introducing hidden distributions), and Neo detects the overfitting of attacks to the leaked model used in its generation. We show that across a variety of tasks and attack methods, Neo is able to filter out attacks from leaked models with very high accuracy, and provides strong protection (7--10 recoveries) against attackers who repeatedly breach the server. Neo performs well against a variety of strong adaptive attacks, dropping slightly in # of breaches recoverable, and demonstrates potential as a complement to DNN defenses in the wild.



## **13. Gradient Concealment: Free Lunch for Defending Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10617v1)

**Authors**: Sen Pei, Jiaxi Sun, Xiaopeng Zhang, Gaofeng Meng

**Abstracts**: Recent studies show that the deep neural networks (DNNs) have achieved great success in various tasks. However, even the \emph{state-of-the-art} deep learning based classifiers are extremely vulnerable to adversarial examples, resulting in sharp decay of discrimination accuracy in the presence of enormous unknown attacks. Given the fact that neural networks are widely used in the open world scenario which can be safety-critical situations, mitigating the adversarial effects of deep learning methods has become an urgent need. Generally, conventional DNNs can be attacked with a dramatically high success rate since their gradient is exposed thoroughly in the white-box scenario, making it effortless to ruin a well trained classifier with only imperceptible perturbations in the raw data space. For tackling this problem, we propose a plug-and-play layer that is training-free, termed as \textbf{G}radient \textbf{C}oncealment \textbf{M}odule (GCM), concealing the vulnerable direction of gradient while guaranteeing the classification accuracy during the inference time. GCM reports superior defense results on the ImageNet classification benchmark, improving up to 63.41\% top-1 attack robustness (AR) when faced with adversarial inputs compared to the vanilla DNNs. Moreover, we use GCM in the CVPR 2022 Robust Classification Challenge, currently achieving \textbf{2nd} place in Phase II with only a tiny version of ConvNext. The code will be made available.



## **14. SERVFAIL: The Unintended Consequences of Algorithm Agility in DNSSEC**

cs.CR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10608v1)

**Authors**: Elias Heftrig, Jean-Pierre Seifert, Haya Shulman, Peter Thomassen, Michael Waidner, Nils Wisiol

**Abstracts**: Cryptographic algorithm agility is an important property for DNSSEC: it allows easy deployment of new algorithms if the existing ones are no longer secure. Significant operational and research efforts are dedicated to pushing the deployment of new algorithms in DNSSEC forward. Recent research shows that DNSSEC is gradually achieving algorithm agility: most DNSSEC supporting resolvers can validate a number of different algorithms and domains are increasingly signed with cryptographically strong ciphers.   In this work we show for the first time that the cryptographic agility in DNSSEC, although critical for making DNS secure with strong cryptography, also introduces a severe vulnerability. We find that under certain conditions, when new algorithms are listed in signed DNS responses, the resolvers do not validate DNSSEC. As a result, domains that deploy new ciphers, risk exposing the validating resolvers to cache poisoning attacks.   We use this to develop DNSSEC-downgrade attacks and show that in some situations these attacks can be launched even by off-path adversaries. We experimentally and ethically evaluate our attacks against popular DNS resolver implementations, public DNS providers, and DNS services used by web clients worldwide. We validate the success of DNSSEC-downgrade attacks by poisoning the resolvers: we inject fake records, in signed domains, into the caches of validating resolvers. We find that major DNS providers, such as Google Public DNS and Cloudflare, as well as 70% of DNS resolvers used by web clients are vulnerable to our attacks.   We trace the factors that led to this situation and provide recommendations.



## **15. On the Feasibility and Generality of Patch-based Adversarial Attacks on Semantic Segmentation Problems**

cs.CV

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10539v1)

**Authors**: Soma Kontar, Andras Horvath

**Abstracts**: Deep neural networks were applied with success in a myriad of applications, but in safety critical use cases adversarial attacks still pose a significant threat. These attacks were demonstrated on various classification and detection tasks and are usually considered general in a sense that arbitrary network outputs can be generated by them.   In this paper we will demonstrate through simple case studies both in simulation and in real-life, that patch based attacks can be utilised to alter the output of segmentation networks. Through a few examples and the investigation of network complexity, we will also demonstrate that the number of possible output maps which can be generated via patch-based attacks of a given size is typically smaller than the area they effect or areas which should be attacked in case of practical applications.   We will prove that based on these results most patch-based attacks cannot be general in practice, namely they can not generate arbitrary output maps or if they could, they are spatially limited and this limit is significantly smaller than the receptive field of the patches.



## **16. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

cs.IR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2204.01321v2)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed.   In this paper, we introduce the Word Substitution Ranking Attack (WSRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but can only acquire the rank positions of the partial retrieved list by querying the target model. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations.   Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.



## **17. Robust Sensible Adversarial Learning of Deep Neural Networks for Image Classification**

cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10457v1)

**Authors**: Jungeum Kim, Xiao Wang

**Abstracts**: The idea of robustness is central and critical to modern statistical analysis. However, despite the recent advances of deep neural networks (DNNs), many studies have shown that DNNs are vulnerable to adversarial attacks. Making imperceptible changes to an image can cause DNN models to make the wrong classification with high confidence, such as classifying a benign mole as a malignant tumor and a stop sign as a speed limit sign. The trade-off between robustness and standard accuracy is common for DNN models. In this paper, we introduce sensible adversarial learning and demonstrate the synergistic effect between pursuits of standard natural accuracy and robustness. Specifically, we define a sensible adversary which is useful for learning a robust model while keeping high natural accuracy. We theoretically establish that the Bayes classifier is the most robust multi-class classifier with the 0-1 loss under sensible adversarial learning. We propose a novel and efficient algorithm that trains a robust model using implicit loss truncation. We apply sensible adversarial learning for large-scale image classification to a handwritten digital image dataset called MNIST and an object recognition colored image dataset called CIFAR10. We have performed an extensive comparative study to compare our method with other competitive methods. Our experiments empirically demonstrate that our method is not sensitive to its hyperparameter and does not collapse even with a small model capacity while promoting robustness against various attacks and keeping high natural accuracy.



## **18. Vulnerability Analysis and Performance Enhancement of Authentication Protocol in Dynamic Wireless Power Transfer Systems**

cs.CR

16 pages, conference

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10292v1)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstracts**: Recent advancements in wireless charging technology, as well as the possibility of utilizing it in the Electric Vehicle (EV) domain for dynamic charging solutions, have fueled the demand for a secure and usable protocol in the Dynamic Wireless Power Transfer (DWPT) technology. The DWPT must operate in the presence of malicious adversaries that can undermine the charging process and harm the customer service quality, while preserving the privacy of the users. Recently, it was shown that the DWPT system is susceptible to adversarial attacks, including replay, denial-of-service and free-riding attacks, which can lead to the adversary blocking the authorized user from charging, enabling free charging for free riders and exploiting the location privacy of the customers. In this paper, we study the current State-Of-The-Art (SOTA) authentication protocols and make the following two contributions: a) we show that the SOTA is vulnerable to the tracking of the user activity and b) we propose an enhanced authentication protocol that eliminates the vulnerability while providing improved efficiency compared to the SOTA authentication protocols. By adopting authentication messages based only on exclusive OR operations, hashing, and hash chains, we optimize the protocol to achieve a complexity that varies linearly with the number of charging pads, providing improved scalability. Compared to SOTA, the proposed scheme has a performance gain in the computational cost of around 90% on average for each pad.



## **19. Adversarial Body Shape Search for Legged Robots**

cs.RO

6 pages, 7 figures

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10187v1)

**Authors**: Takaaki Azakami, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: We propose an evolutionary computation method for an adversarial attack on the length and thickness of parts of legged robots by deep reinforcement learning. This attack changes the robot body shape and interferes with walking-we call the attacked body as adversarial body shape. The evolutionary computation method searches adversarial body shape by minimizing the expected cumulative reward earned through walking simulation. To evaluate the effectiveness of the proposed method, we perform experiments with three-legged robots, Walker2d, Ant-v2, and Humanoid-v2 in OpenAI Gym. The experimental results reveal that Walker2d and Ant-v2 are more vulnerable to the attack on the length than the thickness of the body parts, whereas Humanoid-v2 is vulnerable to the attack on both of the length and thickness. We further identify that the adversarial body shapes break left-right symmetry or shift the center of gravity of the legged robots. Finding adversarial body shape can be used to proactively diagnose the vulnerability of legged robot walking.



## **20. Getting a-Round Guarantees: Floating-Point Attacks on Certified Robustness**

cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10159v1)

**Authors**: Jiankai Jin, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstracts**: Adversarial examples pose a security risk as they can alter a classifier's decision through slight perturbations to a benign input. Certified robustness has been proposed as a mitigation strategy where given an input $x$, a classifier returns a prediction and a radius with a provable guarantee that any perturbation to $x$ within this radius (e.g., under the $L_2$ norm) will not alter the classifier's prediction. In this work, we show that these guarantees can be invalidated due to limitations of floating-point representation that cause rounding errors. We design a rounding search method that can efficiently exploit this vulnerability to find adversarial examples within the certified radius. We show that the attack can be carried out against several linear classifiers that have exact certifiable guarantees and against neural network verifiers that return a certified lower bound on a robust radius. Our experiments demonstrate over 50% attack success rate on random linear classifiers, up to 35% on a breast cancer dataset for logistic regression, and a 9% attack success rate on the MNIST dataset for a neural network whose certified radius was verified by a prominent bound propagation method. We also show that state-of-the-art random smoothed classifiers for neural networks are also susceptible to adversarial examples (e.g., up to 2% attack rate on CIFAR10)-validating the importance of accounting for the error rate of robustness guarantees of such classifiers in practice. Finally, as a mitigation, we advocate the use of rounded interval arithmetic to account for rounding errors.



## **21. Generating Semantic Adversarial Examples via Feature Manipulation**

cs.LG

arXiv admin note: substantial text overlap with arXiv:1705.09064 by  other authors

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2001.02297v2)

**Authors**: Shuo Wang, Surya Nepal, Carsten Rudolph, Marthie Grobler, Shangyu Chen, Tianle Chen

**Abstracts**: The vulnerability of deep neural networks to adversarial attacks has been widely demonstrated (e.g., adversarial example attacks). Traditional attacks perform unstructured pixel-wise perturbation to fool the classifier. An alternative approach is to have perturbations in the latent space. However, such perturbations are hard to control due to the lack of interpretability and disentanglement. In this paper, we propose a more practical adversarial attack by designing structured perturbation with semantic meanings. Our proposed technique manipulates the semantic attributes of images via the disentangled latent codes. The intuition behind our technique is that images in similar domains have some commonly shared but theme-independent semantic attributes, e.g. thickness of lines in handwritten digits, that can be bidirectionally mapped to disentangled latent codes. We generate adversarial perturbation by manipulating a single or a combination of these latent codes and propose two unsupervised semantic manipulation approaches: vector-based disentangled representation and feature map-based disentangled representation, in terms of the complexity of the latent codes and smoothness of the reconstructed images. We conduct extensive experimental evaluations on real-world image data to demonstrate the power of our attacks for black-box classifiers. We further demonstrate the existence of a universal, image-agnostic semantic adversarial example.



## **22. Adversarial joint attacks on legged robots**

cs.RO

6 pages, 8 figures

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10098v1)

**Authors**: Takuto Otomo, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: We address adversarial attacks on the actuators at the joints of legged robots trained by deep reinforcement learning. The vulnerability to the joint attacks can significantly impact the safety and robustness of legged robots. In this study, we demonstrate that the adversarial perturbations to the torque control signals of the actuators can significantly reduce the rewards and cause walking instability in robots. To find the adversarial torque perturbations, we develop black-box adversarial attacks, where, the adversary cannot access the neural networks trained by deep reinforcement learning. The black box attack can be applied to legged robots regardless of the architecture and algorithms of deep reinforcement learning. We employ three search methods for the black-box adversarial attacks: random search, differential evolution, and numerical gradient descent methods. In experiments with the quadruped robot Ant-v2 and the bipedal robot Humanoid-v2, in OpenAI Gym environments, we find that differential evolution can efficiently find the strongest torque perturbations among the three methods. In addition, we realize that the quadruped robot Ant-v2 is vulnerable to the adversarial perturbations, whereas the bipedal robot Humanoid-v2 is robust to the perturbations. Consequently, the joint attacks can be used for proactive diagnosis of robot walking instability.



## **23. SafeNet: Mitigating Data Poisoning Attacks on Private Machine Learning**

cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.09986v1)

**Authors**: Harsh Chaudhari, Matthew Jagielski, Alina Oprea

**Abstracts**: Secure multiparty computation (MPC) has been proposed to allow multiple mutually distrustful data owners to jointly train machine learning (ML) models on their combined data. However, the datasets used for training ML models might be under the control of an adversary mounting a data poisoning attack, and MPC prevents inspecting training sets to detect poisoning. We show that multiple MPC frameworks for private ML training are susceptible to backdoor and targeted poisoning attacks. To mitigate this, we propose SafeNet, a framework for building ensemble models in MPC with formal guarantees of robustness to data poisoning attacks. We extend the security definition of private ML training to account for poisoning and prove that our SafeNet design satisfies the definition. We demonstrate SafeNet's efficiency, accuracy, and resilience to poisoning on several machine learning datasets and models. For instance, SafeNet reduces backdoor attack success from 100% to 0% for a neural network model, while achieving 39x faster training and 36x less communication than the four-party MPC framework of Dalskov et al.



## **24. Adversarial Sample Detection for Speaker Verification by Neural Vocoders**

cs.SD

Accepted by ICASSP 2022

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2107.00309v4)

**Authors**: Haibin Wu, Po-chun Hsu, Ji Gao, Shanshan Zhang, Shen Huang, Jian Kang, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Automatic speaker verification (ASV), one of the most important technology for biometric identification, has been widely adopted in security-critical applications. However, ASV is seriously vulnerable to recently emerged adversarial attacks, yet effective countermeasures against them are limited. In this paper, we adopt neural vocoders to spot adversarial samples for ASV. We use the neural vocoder to re-synthesize audio and find that the difference between the ASV scores for the original and re-synthesized audio is a good indicator for discrimination between genuine and adversarial samples. This effort is, to the best of our knowledge, among the first to pursue such a technical direction for detecting time-domain adversarial samples for ASV, and hence there is a lack of established baselines for comparison. Consequently, we implement the Griffin-Lim algorithm as the detection baseline. The proposed approach achieves effective detection performance that outperforms the baselines in all the settings. We also show that the neural vocoder adopted in the detection framework is dataset-independent. Our codes will be made open-source for future works to do fair comparison.



## **25. Focused Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09624v1)

**Authors**: Thomas Cilloni, Charles Walter, Charles Fleming

**Abstracts**: Recent advances in machine learning show that neural models are vulnerable to minimally perturbed inputs, or adversarial examples. Adversarial algorithms are optimization problems that minimize the accuracy of ML models by perturbing inputs, often using a model's loss function to craft such perturbations. State-of-the-art object detection models are characterized by very large output manifolds due to the number of possible locations and sizes of objects in an image. This leads to their outputs being sparse and optimization problems that use them incur a lot of unnecessary computation.   We propose to use a very limited subset of a model's learned manifold to compute adversarial examples. Our \textit{Focused Adversarial Attacks} (FA) algorithm identifies a small subset of sensitive regions to perform gradient-based adversarial attacks. FA is significantly faster than other gradient-based attacks when a model's manifold is sparsely activated. Also, its perturbations are more efficient than other methods under the same perturbation constraints. We evaluate FA on the COCO 2017 and Pascal VOC 2007 detection datasets.



## **26. Improving Robustness against Real-World and Worst-Case Distribution Shifts through Decision Region Quantification**

cs.LG

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09619v1)

**Authors**: Leo Schwinn, Leon Bungert, An Nguyen, René Raab, Falk Pulsmeyer, Doina Precup, Björn Eskofier, Dario Zanca

**Abstracts**: The reliability of neural networks is essential for their use in safety-critical applications. Existing approaches generally aim at improving the robustness of neural networks to either real-world distribution shifts (e.g., common corruptions and perturbations, spatial transformations, and natural adversarial examples) or worst-case distribution shifts (e.g., optimized adversarial examples). In this work, we propose the Decision Region Quantification (DRQ) algorithm to improve the robustness of any differentiable pre-trained model against both real-world and worst-case distribution shifts in the data. DRQ analyzes the robustness of local decision regions in the vicinity of a given data point to make more reliable predictions. We theoretically motivate the DRQ algorithm by showing that it effectively smooths spurious local extrema in the decision surface. Furthermore, we propose an implementation using targeted and untargeted adversarial attacks. An extensive empirical evaluation shows that DRQ increases the robustness of adversarially and non-adversarially trained models against real-world and worst-case distribution shifts on several computer vision benchmark datasets.



## **27. Transferable Physical Attack against Object Detection with Separable Attention**

cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09592v1)

**Authors**: Yu Zhang, Zhiqiang Gong, Yichuang Zhang, YongQian Li, Kangcheng Bin, Jiahao Qi, Wei Xue, Ping Zhong

**Abstracts**: Transferable adversarial attack is always in the spotlight since deep learning models have been demonstrated to be vulnerable to adversarial samples. However, existing physical attack methods do not pay enough attention on transferability to unseen models, thus leading to the poor performance of black-box attack.In this paper, we put forward a novel method of generating physically realizable adversarial camouflage to achieve transferable attack against detection models. More specifically, we first introduce multi-scale attention maps based on detection models to capture features of objects with various resolutions. Meanwhile, we adopt a sequence of composite transformations to obtain the averaged attention maps, which could curb model-specific noise in the attention and thus further boost transferability. Unlike the general visualization interpretation methods where model attention should be put on the foreground object as much as possible, we carry out attack on separable attention from the opposite perspective, i.e. suppressing attention of the foreground and enhancing that of the background. Consequently, transferable adversarial camouflage could be yielded efficiently with our novel attention-based loss function. Extensive comparison experiments verify the superiority of our method to state-of-the-art methods.



## **28. On Trace of PGD-Like Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09586v1)

**Authors**: Mo Zhou, Vishal M. Patel

**Abstracts**: Adversarial attacks pose safety and security concerns for deep learning applications. Yet largely imperceptible, a strong PGD-like attack may leave strong trace in the adversarial example. Since attack triggers the local linearity of a network, we speculate network behaves in different extents of linearity for benign examples and adversarial examples. Thus, we construct Adversarial Response Characteristics (ARC) features to reflect the model's gradient consistency around the input to indicate the extent of linearity. Under certain conditions, it shows a gradually varying pattern from benign example to adversarial example, as the later leads to Sequel Attack Effect (SAE). ARC feature can be used for informed attack detection (perturbation magnitude is known) with binary classifier, or uninformed attack detection (perturbation magnitude is unknown) with ordinal regression. Due to the uniqueness of SAE to PGD-like attacks, ARC is also capable of inferring other attack details such as loss function, or the ground-truth label as a post-processing defense. Qualitative and quantitative evaluations manifest the effectiveness of ARC feature on CIFAR-10 w/ ResNet-18 and ImageNet w/ ResNet-152 and SwinT-B-IN1K with considerable generalization among PGD-like attacks despite domain shift. Our method is intuitive, light-weighted, non-intrusive, and data-undemanding.



## **29. Defending Against Adversarial Attacks by Energy Storage Facility**

cs.CR

arXiv admin note: text overlap with arXiv:1904.06606 by other authors

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09522v1)

**Authors**: Jiawei Li, Jianxiao Wang, Lin Chen, Yang Yu

**Abstracts**: Adversarial attacks on data-driven algorithms applied in pow-er system will be a new type of threat on grid security. Litera-ture has demonstrated the adversarial attack on deep-neural network can significantly misleading the load forecast of a power system. However, it is unclear how the new type of at-tack impact on the operation of grid system. In this research, we manifest that the adversarial algorithm attack induces a significant cost-increase risk which will be exacerbated by the growing penetration of intermittent renewable energy. In Texas, a 5% adversarial attack can increase the total generation cost by 17% in a quarter, which account for around 20 million dollars. When wind-energy penetration increases to over 40%, the 5% adver-sarial attack will inflate the generation cost by 23%. Our re-search discovers a novel approach of defending against the adversarial attack: investing on energy-storage system. All current literature focuses on developing algorithm to defending against adversarial attack. We are the first research revealing the capability of using facility in physical system to defending against the adversarial algorithm attack in a system of Internet of Thing, such as smart grid system.



## **30. Enhancing the Transferability of Adversarial Examples via a Few Queries**

cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09518v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Due to the vulnerability of deep neural networks, the black-box attack has drawn great attention from the community. Though transferable priors decrease the query number of the black-box query attacks in recent efforts, the average number of queries is still larger than 100, which is easily affected by the number of queries limit policy. In this work, we propose a novel method called query prior-based method to enhance the family of fast gradient sign methods and improve their attack transferability by using a few queries. Specifically, for the untargeted attack, we find that the successful attacked adversarial examples prefer to be classified as the wrong categories with higher probability by the victim model. Therefore, the weighted augmented cross-entropy loss is proposed to reduce the gradient angle between the surrogate model and the victim model for enhancing the transferability of the adversarial examples. Theoretical analysis and extensive experiments demonstrate that our method could significantly improve the transferability of gradient-based adversarial attacks on CIFAR10/100 and ImageNet and outperform the black-box query attack with the same few queries.



## **31. GUARD: Graph Universal Adversarial Defense**

cs.LG

Preprint. Code is publicly available at  https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2204.09803v2)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Jiawang Dan, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named Graph Universal Adversarial Defense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms state-of-the-art defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.



## **32. Sparse Adversarial Attack in Multi-agent Reinforcement Learning**

cs.AI

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09362v1)

**Authors**: Yizheng Hu, Zhihua Zhang

**Abstracts**: Cooperative multi-agent reinforcement learning (cMARL) has many real applications, but the policy trained by existing cMARL algorithms is not robust enough when deployed. There exist also many methods about adversarial attacks on the RL system, which implies that the RL system can suffer from adversarial attacks, but most of them focused on single agent RL. In this paper, we propose a \textit{sparse adversarial attack} on cMARL systems. We use (MA)RL with regularization to train the attack policy. Our experiments show that the policy trained by the current cMARL algorithm can obtain poor performance when only one or a few agents in the team (e.g., 1 of 8 or 5 of 25) were attacked at a few timesteps (e.g., attack 3 of total 40 timesteps).



## **33. Backdoor Attacks on Bayesian Neural Networks using Reverse Distribution**

cs.CR

9 pages, 7 figures

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.09167v1)

**Authors**: Zhixin Pan, Prabhat Mishra

**Abstracts**: Due to cost and time-to-market constraints, many industries outsource the training process of machine learning models (ML) to third-party cloud service providers, popularly known as ML-asa-Service (MLaaS). MLaaS creates opportunity for an adversary to provide users with backdoored ML models to produce incorrect predictions only in extremely rare (attacker-chosen) scenarios. Bayesian neural networks (BNN) are inherently immune against backdoor attacks since the weights are designed to be marginal distributions to quantify the uncertainty. In this paper, we propose a novel backdoor attack based on effective learning and targeted utilization of reverse distribution. This paper makes three important contributions. (1) To the best of our knowledge, this is the first backdoor attack that can effectively break the robustness of BNNs. (2) We produce reverse distributions to cancel the original distributions when the trigger is activated. (3) We propose an efficient solution for merging probability distributions in BNNs. Experimental results on diverse benchmark datasets demonstrate that our proposed attack can achieve the attack success rate (ASR) of 100%, while the ASR of the state-of-the-art attacks is lower than 60%.



## **34. VLC Physical Layer Security through RIS-aided Jamming Receiver for 6G Wireless Networks**

cs.CR

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.09026v1)

**Authors**: Simone Soderi, Alessandro Brighente, Federico Turrin, Mauro Conti

**Abstracts**: Visible Light Communication (VLC) is one the most promising enabling technology for future 6G networks to overcome Radio-Frequency (RF)-based communication limitations thanks to a broader bandwidth, higher data rate, and greater efficiency. However, from the security perspective, VLCs suffer from all known wireless communication security threats (e.g., eavesdropping and integrity attacks). For this reason, security researchers are proposing innovative Physical Layer Security (PLS) solutions to protect such communication. Among the different solutions, the novel Reflective Intelligent Surface (RIS) technology coupled with VLCs has been successfully demonstrated in recent work to improve the VLC communication capacity. However, to date, the literature still lacks analysis and solutions to show the PLS capability of RIS-based VLC communication. In this paper, we combine watermarking and jamming primitives through the Watermark Blind Physical Layer Security (WBPLSec) algorithm to secure VLC communication at the physical layer. Our solution leverages RIS technology to improve the security properties of the communication. By using an optimization framework, we can calculate RIS phases to maximize the WBPLSec jamming interference schema over a predefined area in the room. In particular, compared to a scenario without RIS, our solution improves the performance in terms of secrecy capacity without any assumption about the adversary's location. We validate through numerical evaluations the positive impact of RIS-aided solution to increase the secrecy capacity of the legitimate jamming receiver in a VLC indoor scenario. Our results show that the introduction of RIS technology extends the area where secure communication occurs and that by increasing the number of RIS elements the outage probability decreases.



## **35. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

cs.CV

13 pages

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2005.09147v8)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial noises. By adding adversarial noises to training samples, adversarial training can improve the model's robustness against adversarial noises. However, adversarial training samples with excessive noises can harm standard accuracy, which may be unacceptable for many medical image analysis applications. This issue has been termed the trade-off between standard accuracy and adversarial robustness. In this paper, we hypothesize that this issue may be alleviated if the adversarial samples for training are placed right on the decision boundaries. Based on this hypothesis, we design an adaptive adversarial training method, named IMA. For each individual training sample, IMA makes a sample-wise estimation of the upper bound of the adversarial perturbation. In the training process, each of the sample-wise adversarial perturbations is gradually increased to match the margin. Once an equilibrium state is reached, the adversarial perturbations will stop increasing. IMA is evaluated on publicly available datasets under two popular adversarial attacks, PGD and IFGSM. The results show that: (1) IMA significantly improves adversarial robustness of DNN classifiers, which achieves the state-of-the-art performance; (2) IMA has a minimal reduction in clean accuracy among all competing defense methods; (3) IMA can be applied to pretrained models to reduce time cost; (4) IMA can be applied to the state-of-the-art medical image segmentation networks, with outstanding performance. We hope our work may help to lift the trade-off between adversarial robustness and clean accuracy and facilitate the development of robust applications in the medical field. The source code will be released when this paper is published.



## **36. Property Unlearning: A Defense Strategy Against Property Inference Attacks**

cs.CR

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.08821v1)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstracts**: During the training of machine learning models, they may store or "learn" more information about the training data than what is actually needed for the prediction or classification task. This is exploited by property inference attacks which aim at extracting statistical properties from the training data of a given model without having access to the training data itself. These properties may include the quality of pictures to identify the camera model, the age distribution to reveal the target audience of a product, or the included host types to refine a malware attack in computer networks. This attack is especially accurate when the attacker has access to all model parameters, i.e., in a white-box scenario. By defending against such attacks, model owners are able to ensure that their training data, associated properties, and thus their intellectual property stays private, even if they deliberately share their models, e.g., to train collaboratively, or if models are leaked. In this paper, we introduce property unlearning, an effective defense mechanism against white-box property inference attacks, independent of the training data type, model task, or number of properties. Property unlearning mitigates property inference attacks by systematically changing the trained weights and biases of a target model such that an adversary cannot extract chosen properties. We empirically evaluate property unlearning on three different data sets, including tabular and image data, and two types of artificial neural networks. Our results show that property unlearning is both efficient and reliable to protect machine learning models against property inference attacks, with a good privacy-utility trade-off. Furthermore, our approach indicates that this mechanism is also effective to unlearn multiple properties.



## **37. Passive Defense Against 3D Adversarial Point Clouds Through the Lens of 3D Steganalysis**

cs.MM

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.08738v1)

**Authors**: Jiahao Zhu

**Abstracts**: Nowadays, 3D data plays an indelible role in the computer vision field. However, extensive studies have proved that deep neural networks (DNNs) fed with 3D data, such as point clouds, are susceptible to adversarial examples, which aim to misguide DNNs and might bring immeasurable losses. Currently, 3D adversarial point clouds are chiefly generated in three fashions, i.e., point shifting, point adding, and point dropping. These point manipulations would modify geometrical properties and local correlations of benign point clouds more or less. Motivated by this basic fact, we propose to defend such adversarial examples with the aid of 3D steganalysis techniques. Specifically, we first introduce an adversarial attack and defense model adapted from the celebrated Prisoners' Problem in steganography to help us comprehend 3D adversarial attack and defense more generally. Then we rethink two significant but vague concepts in the field of adversarial example, namely, active defense and passive defense, from the perspective of steganalysis. Most importantly, we design a 3D adversarial point cloud detector through the lens of 3D steganalysis. Our detector is double-blind, that is to say, it does not rely on the exact knowledge of the adversarial attack means and victim models. To enable the detector to effectively detect malicious point clouds, we craft a 64-D discriminant feature set, including features related to first-order and second-order local descriptions of point clouds. To our knowledge, this work is the first to apply 3D steganalysis to 3D adversarial example defense. Extensive experimental results demonstrate that the proposed 3D adversarial point cloud detector can achieve good detection performance on multiple types of 3D adversarial point clouds.



## **38. Policy Distillation with Selective Input Gradient Regularization for Efficient Interpretability**

cs.LG

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.08685v1)

**Authors**: Jinwei Xing, Takashi Nagata, Xinyun Zou, Emre Neftci, Jeffrey L. Krichmar

**Abstracts**: Although deep Reinforcement Learning (RL) has proven successful in a wide range of tasks, one challenge it faces is interpretability when applied to real-world problems. Saliency maps are frequently used to provide interpretability for deep neural networks. However, in the RL domain, existing saliency map approaches are either computationally expensive and thus cannot satisfy the real-time requirement of real-world scenarios or cannot produce interpretable saliency maps for RL policies. In this work, we propose an approach of Distillation with selective Input Gradient Regularization (DIGR) which uses policy distillation and input gradient regularization to produce new policies that achieve both high interpretability and computation efficiency in generating saliency maps. Our approach is also found to improve the robustness of RL policies to multiple adversarial attacks. We conduct experiments on three tasks, MiniGrid (Fetch Object), Atari (Breakout) and CARLA Autonomous Driving, to demonstrate the importance and effectiveness of our approach.



## **39. Longest Chain Consensus Under Bandwidth Constraint**

cs.CR

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2111.12332v3)

**Authors**: Joachim Neu, Srivatsan Sridhar, Lei Yang, David Tse, Mohammad Alizadeh

**Abstracts**: Spamming attacks are a serious concern for consensus protocols, as witnessed by recent outages of a major blockchain, Solana. They cause congestion and excessive message delays in a real network due to its bandwidth constraints. In contrast, longest chain (LC), an important family of consensus protocols, has previously only been proven secure assuming an idealized network model in which all messages are delivered within bounded delay. This model-reality mismatch is further aggravated for Proof-of-Stake (PoS) LC where the adversary can spam the network with equivocating blocks. Hence, we extend the network model to capture bandwidth constraints, under which nodes now need to choose carefully which blocks to spend their limited download budget on. To illustrate this point, we show that 'download along the longest header chain', a natural download rule for Proof-of-Work (PoW) LC, is insecure for PoS LC. We propose a simple rule 'download towards the freshest block', formalize two common heuristics 'not downloading equivocations' and 'blocklisting', and prove in a unified framework that PoS LC with any one of these download rules is secure in bandwidth-constrained networks. In experiments, we validate our claims and showcase the behavior of these download rules under attack. By composing multiple instances of a PoS LC protocol with a suitable download rule in parallel, we obtain a PoS consensus protocol that achieves a constant fraction of the network's throughput limit even under worst-case adversarial strategies.



## **40. An Integrated Approach for Energy Efficient Handover and Key Distribution Protocol for Secure NC-enabled Small Cells**

cs.NI

Preprint of the paper accepted at Computer Networks

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08641v1)

**Authors**: Vipindev Adat Vasudevan, Muhammad Tayyab, George P. Koudouridis, Xavier Gelabert, Ilias Politis

**Abstracts**: Future wireless networks must serve dense mobile networks with high data rates, keeping energy requirements to a possible minimum. The small cell-based network architecture and device-to-device (D2D) communication are already being considered part of 5G networks and beyond. In such environments, network coding (NC) can be employed to achieve both higher throughput and energy efficiency. However, NC-enabled systems need to address security challenges specific to NC, such as pollution attacks. All integrity schemes against pollution attacks generally require proper key distribution and management to ensure security in a mobile environment. Additionally, the mobility requirements in small cell environments are more challenging and demanding in terms of signaling overhead. This paper proposes a blockchain-assisted key distribution protocol tailored for MAC-based integrity schemes, which combined with an uplink reference signal (UL RS) handover mechanism, enables energy efficient secure NC. The performance analysis of the protocol during handover scenarios indicates its suitability for ensuring high level of security against pollution attacks in dense small cell environments with multiple adversaries being present. Furthermore, the proposed scheme achieves lower bandwidth and signaling overhead during handover compared to legacy schemes and the signaling cost reduces significantly as the communication progresses, thus enhancing the network's cumulative energy efficiency.



## **41. Hierarchical Distribution-Aware Testing of Deep Learning**

cs.SE

Under Review

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08589v1)

**Authors**: Wei Huang, Xingyu Zhao, Alec Banks, Victoria Cox, Xiaowei Huang

**Abstracts**: With its growing use in safety/security-critical applications, Deep Learning (DL) has raised increasing concerns regarding its dependability. In particular, DL has a notorious problem of lacking robustness. Despite recent efforts made in detecting Adversarial Examples (AEs) via state-of-the-art attacking and testing methods, they are normally input distribution agnostic and/or disregard the perception quality of AEs. Consequently, the detected AEs are irrelevant inputs in the application context or unnatural/unrealistic that can be easily noticed by humans. This may lead to a limited effect on improving the DL model's dependability, as the testing budget is likely to be wasted on detecting AEs that are encountered very rarely in its real-life operations. In this paper, we propose a new robustness testing approach for detecting AEs that considers both the input distribution and the perceptual quality of inputs. The two considerations are encoded by a novel hierarchical mechanism. First, at the feature level, the input data distribution is extracted and approximated by data compression techniques and probability density estimators. Such quantified feature level distribution, together with indicators that are highly correlated with local robustness, are considered in selecting test seeds. Given a test seed, we then develop a two-step genetic algorithm for local test case generation at the pixel level, in which two fitness functions work alternatively to control the quality of detected AEs. Finally, extensive experiments confirm that our holistic approach considering hierarchical distributions at feature and pixel levels is superior to state-of-the-arts that either disregard any input distribution or only consider a single (non-hierarchical) distribution, in terms of not only the quality of detected AEs but also improving the overall robustness of the DL model under testing.



## **42. F3B: A Low-Latency Commit-and-Reveal Architecture to Mitigate Blockchain Front-Running**

cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08529v1)

**Authors**: Haoqian Zhang, Louis-Henri Merino, Vero Estrada-Galinanes, Bryan Ford

**Abstracts**: Front-running attacks, which benefit from advanced knowledge of pending transactions, have proliferated in the cryptocurrency space since the emergence of decentralized finance. Front-running causes devastating losses to honest participants$\unicode{x2013}$estimated at \$280M each month$\unicode{x2013}$and endangers the fairness of the ecosystem. We present Flash Freezing Flash Boys (F3B), a blockchain architecture to address front-running attacks by relying on a commit-and-reveal scheme where the contents of transactions are encrypted and later revealed by a decentralized secret-management committee once the underlying consensus layer has committed the transaction. F3B mitigates front-running attacks because an adversary can no longer read the content of a transaction before commitment, thus preventing the adversary from benefiting from advance knowledge of pending transactions. We design F3B to be agnostic to the underlying consensus algorithm and compatible with legacy smart contracts by addressing front-running at the blockchain architecture level. Unlike existing commit-and-reveal approaches, F3B only requires writing data onto the underlying blockchain once, establishing a significant overhead reduction. An exploration of F3B shows that with a secret-management committee consisting of 8 and 128 members, F3B presents between 0.1 and 1.8 seconds of transaction-processing latency, respectively.



## **43. On the Privacy of Decentralized Machine Learning**

cs.CR

17 pages

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08443v1)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstracts**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at circumventing the main limitations of federated learning. We identify the decentralized learning properties that affect users' privacy and we introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantages over more practical approaches such as federated learning. Rather, it tends to degrade users' privacy by increasing the attack surface and enabling any user in the system to perform powerful privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also reveal that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require abandoning any possible advantage over the federated setup, completely defeating the objective of the decentralized approach.



## **44. Can You Still See Me?: Reconstructing Robot Operations Over End-to-End Encrypted Channels**

cs.CR

13 pages, 7 figures, poster presented at wisec'22

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08426v1)

**Authors**: Ryan Shah, Chuadhry Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected robots play a key role in Industry 4.0, providing automation and higher efficiency for many industrial workflows. Unfortunately, these robots can leak sensitive information regarding these operational workflows to remote adversaries. While there exists mandates for the use of end-to-end encryption for data transmission in such settings, it is entirely possible for passive adversaries to fingerprint and reconstruct entire workflows being carried out -- establishing an understanding of how facilities operate. In this paper, we investigate whether a remote attacker can accurately fingerprint robot movements and ultimately reconstruct operational workflows. Using a neural network approach to traffic analysis, we find that one can predict TLS-encrypted movements with around \textasciitilde60\% accuracy, increasing to near-perfect accuracy under realistic network conditions. Further, we also find that attackers can reconstruct warehousing workflows with similar success. Ultimately, simply adopting best cybersecurity practices is clearly not enough to stop even weak (passive) adversaries.



## **45. Bankrupting DoS Attackers Despite Uncertainty**

cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08287v1)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstracts**: On-demand provisioning in the cloud allows for services to remain available despite massive denial-of-service (DoS) attacks. Unfortunately, on-demand provisioning is expensive and must be weighed against the costs incurred by an adversary. This leads to a recent threat known as economic denial-of-sustainability (EDoS), where the cost for defending a service is higher than that of attacking.   A natural approach for combating EDoS is to impose costs via resource burning (RB). Here, a client must verifiably consume resources -- for example, by solving a computational challenge -- before service is rendered. However, prior approaches with security guarantees do not account for the cost on-demand provisioning.   Another valuable defensive tool is to use a classifier in order to discern good jobs from a legitimate client, versus bad jobs from the adversary. However, while useful, uncertainty arises from classification error, which still allows bad jobs to consume server resources. Thus, classification is not a solution by itself.   Here, we propose an EDoS defense, RootDef, that leverages both RB and classification, while accounting for both the costs of resource burning and on-demand provisioning. Specifically, against an adversary that expends $B$ resources to attack, the total cost for defending is $\tilde{O}( \sqrt{B\,g} + B^{2/3} + g)$, where $g$ is the number of good jobs and $\tilde{O}$ refers to hidden logarithmic factors in the total number of jobs $n$. Notably, for large $B$ relative to $g$, the adversary has higher cost, implying that the algorithm has an economic advantage. Finally, we prove a lower bound showing that RootDef has total costs that are asymptotically tight up to logarithmic factors in $n$.



## **46. How Not to Handle Keys: Timing Attacks on FIDO Authenticator Privacy**

cs.CR

to be published in the 22nd Privacy Enhancing Technologies Symposium  (PETS 2022)

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08071v1)

**Authors**: Michal Kepkowski, Lucjan Hanzlik, Ian Wood, Mohamed Ali Kaafar

**Abstracts**: This paper presents a timing attack on the FIDO2 (Fast IDentity Online) authentication protocol that allows attackers to link user accounts stored in vulnerable authenticators, a serious privacy concern. FIDO2 is a new standard specified by the FIDO industry alliance for secure token online authentication. It complements the W3C WebAuthn specification by providing means to use a USB token or other authenticator as a second factor during the authentication process. From a cryptographic perspective, the protocol is a simple challenge-response where the elliptic curve digital signature algorithm is used to sign challenges. To protect the privacy of the user the token uses unique key pairs per service. To accommodate for small memory, tokens use various techniques that make use of a special parameter called a key handle sent by the service to the token. We identify and analyse a vulnerability in the way the processing of key handles is implemented that allows attackers to remotely link user accounts on multiple services. We show that for vulnerable authenticators there is a difference between the time it takes to process a key handle for a different service but correct authenticator, and for a different authenticator but correct service. This difference can be used to perform a timing attack allowing an adversary to link user's accounts across services. We present several real world examples of adversaries that are in a position to execute our attack and can benefit from linking accounts. We found that two of the eight hardware authenticators we tested were vulnerable despite FIDO level 1 certification. This vulnerability cannot be easily mitigated on authenticators because, for security reasons, they usually do not allow firmware updates. In addition, we show that due to the way existing browsers implement the WebAuthn standard, the attack can be executed remotely.



## **47. User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning**

cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2204.02500v2)

**Authors**: Tiantian Feng, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: Many existing privacy-enhanced speech emotion recognition (SER) frameworks focus on perturbing the original speech data through adversarial training within a centralized machine learning setup. However, this privacy protection scheme can fail since the adversary can still access the perturbed data. In recent years, distributed learning algorithms, especially federated learning (FL), have gained popularity to protect privacy in machine learning applications. While FL provides good intuition to safeguard privacy by keeping the data on local devices, prior work has shown that privacy attacks, such as attribute inference attacks, are achievable for SER systems trained using FL. In this work, we propose to evaluate the user-level differential privacy (UDP) in mitigating the privacy leaks of the SER system in FL. UDP provides theoretical privacy guarantees with privacy parameters $\epsilon$ and $\delta$. Our results show that the UDP can effectively decrease attribute information leakage while keeping the utility of the SER system with the adversary accessing one model update. However, the efficacy of the UDP suffers when the FL system leaks more model updates to the adversary. We make the code publicly available to reproduce the results in https://github.com/usc-sail/fed-ser-leakage.



## **48. RoVISQ: Reduction of Video Service Quality via Adversarial Attacks on Deep Learning-based Video Compression**

cs.CV

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2203.10183v2)

**Authors**: Jung-Woo Chang, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstracts**: Video compression plays a crucial role in video streaming and classification systems by maximizing the end-user quality of experience (QoE) at a given bandwidth budget. In this paper, we conduct the first systematic study for adversarial attacks on deep learning-based video compression and downstream classification systems. Our attack framework, dubbed RoVISQ, manipulates the Rate-Distortion (R-D) relationship of a video compression model to achieve one or both of the following goals: (1) increasing the network bandwidth, (2) degrading the video quality for end-users. We further devise new objectives for targeted and untargeted attacks to a downstream video classification service. Finally, we design an input-invariant perturbation that universally disrupts video compression and classification systems in real time. Unlike previously proposed attacks on video classification, our adversarial perturbations are the first to withstand compression. We empirically show the resilience of RoVISQ attacks against various defenses, i.e., adversarial training, video denoising, and JPEG compression. Our extensive experimental results on various video datasets show RoVISQ attacks deteriorate peak signal-to-noise ratio by up to 5.6dB and the bit-rate by up to 2.4 times while achieving over 90% attack success rate on a downstream classifier.



## **49. Classification Auto-Encoder based Detector against Diverse Data Poisoning Attacks**

cs.LG

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2108.04206v2)

**Authors**: Fereshteh Razmi, Li Xiong

**Abstracts**: Poisoning attacks are a category of adversarial machine learning threats in which an adversary attempts to subvert the outcome of the machine learning systems by injecting crafted data into training data set, thus increasing the machine learning model's test error. The adversary can tamper with the data feature space, data labels, or both, each leading to a different attack strategy with different strengths. Various detection approaches have recently emerged, each focusing on one attack strategy. The Achilles heel of many of these detection approaches is their dependence on having access to a clean, untampered data set. In this paper, we propose CAE, a Classification Auto-Encoder based detector against diverse poisoned data. CAE can detect all forms of poisoning attacks using a combination of reconstruction and classification errors without having any prior knowledge of the attack strategy. We show that an enhanced version of CAE (called CAE+) does not have to employ a clean data set to train the defense model. Our experimental results on three real datasets MNIST, Fashion-MNIST and CIFAR demonstrate that our proposed method can maintain its functionality under up to 30% contaminated data and help the defended SVM classifier to regain its best accuracy.



## **50. Transferability of Adversarial Attacks on Synthetic Speech Detection**

cs.SD

5 pages, submit to Interspeech2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07711v1)

**Authors**: Jiacheng Deng, Shunyi Chen, Li Dong, Diqun Yan, Rangding Wang

**Abstracts**: Synthetic speech detection is one of the most important research problems in audio security. Meanwhile, deep neural networks are vulnerable to adversarial attacks. Therefore, we establish a comprehensive benchmark to evaluate the transferability of adversarial attacks on the synthetic speech detection task. Specifically, we attempt to investigate: 1) The transferability of adversarial attacks between different features. 2) The influence of varying extraction hyperparameters of features on the transferability of adversarial attacks. 3) The effect of clipping or self-padding operation on the transferability of adversarial attacks. By performing these analyses, we summarise the weaknesses of synthetic speech detectors and the transferability behaviours of adversarial attacks, which provide insights for future research. More details can be found at https://gitee.com/djc_QRICK/Attack-Transferability-On-Synthetic-Detection.



