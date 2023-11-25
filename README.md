# Latest Adversarial Attack Papers
**update at 2023-11-25 10:54:20**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Adversarial Backdoor Attack by Naturalistic Data Poisoning on Trajectory Prediction in Autonomous Driving**

cs.CV

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2306.15755v2) [paper-pdf](http://arxiv.org/pdf/2306.15755v2)

**Authors**: Mozhgan Pourkeshavarz, Mohammad Sabokrou, Amir Rasouli

**Abstract**: In autonomous driving, behavior prediction is fundamental for safe motion planning, hence the security and robustness of prediction models against adversarial attacks are of paramount importance. We propose a novel adversarial backdoor attack against trajectory prediction models as a means of studying their potential vulnerabilities. Our attack affects the victim at training time via naturalistic, hence stealthy, poisoned samples crafted using a novel two-step approach. First, the triggers are crafted by perturbing the trajectory of attacking vehicle and then disguised by transforming the scene using a bi-level optimization technique. The proposed attack does not depend on a particular model architecture and operates in a black-box manner, thus can be effective without any knowledge of the victim model. We conduct extensive empirical studies using state-of-the-art prediction models on two benchmark datasets using metrics customized for trajectory prediction. We show that the proposed attack is highly effective, as it can significantly hinder the performance of prediction models, unnoticeable by the victims, and efficient as it forces the victim to generate malicious behavior even under constrained conditions. Via ablative studies, we analyze the impact of different attack design choices followed by an evaluation of existing defence mechanisms against the proposed attack.



## **2. Transfer Attacks and Defenses for Large Language Models on Coding Tasks**

cs.LG

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13445v1) [paper-pdf](http://arxiv.org/pdf/2311.13445v1)

**Authors**: Chi Zhang, Zifan Wang, Ravi Mangal, Matt Fredrikson, Limin Jia, Corina Pasareanu

**Abstract**: Modern large language models (LLMs), such as ChatGPT, have demonstrated impressive capabilities for coding tasks including writing and reasoning about code. They improve upon previous neural network models of code, such as code2seq or seq2seq, that already demonstrated competitive results when performing tasks such as code summarization and identifying code vulnerabilities. However, these previous code models were shown vulnerable to adversarial examples, i.e. small syntactic perturbations that do not change the program's semantics, such as the inclusion of "dead code" through false conditions or the addition of inconsequential print statements, designed to "fool" the models. LLMs can also be vulnerable to the same adversarial perturbations but a detailed study on this concern has been lacking so far. In this paper we aim to investigate the effect of adversarial perturbations on coding tasks with LLMs. In particular, we study the transferability of adversarial examples, generated through white-box attacks on smaller code models, to LLMs. Furthermore, to make the LLMs more robust against such adversaries without incurring the cost of retraining, we propose prompt-based defenses that involve modifying the prompt to include additional information such as examples of adversarially perturbed code and explicit instructions for reversing adversarial perturbations. Our experiments show that adversarial examples obtained with a smaller code model are indeed transferable, weakening the LLMs' performance. The proposed defenses show promise in improving the model's resilience, paving the way to more robust defensive solutions for LLMs in code-related applications.



## **3. From Principle to Practice: Vertical Data Minimization for Machine Learning**

cs.LG

Accepted at IEEE S&P 2024

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.10500v2) [paper-pdf](http://arxiv.org/pdf/2311.10500v2)

**Authors**: Robin Staab, Nikola Jovanović, Mislav Balunović, Martin Vechev

**Abstract**: Aiming to train and deploy predictive models, organizations collect large amounts of detailed client data, risking the exposure of private information in the event of a breach. To mitigate this, policymakers increasingly demand compliance with the data minimization (DM) principle, restricting data collection to only that data which is relevant and necessary for the task. Despite regulatory pressure, the problem of deploying machine learning models that obey DM has so far received little attention. In this work, we address this challenge in a comprehensive manner. We propose a novel vertical DM (vDM) workflow based on data generalization, which by design ensures that no full-resolution client data is collected during training and deployment of models, benefiting client privacy by reducing the attack surface in case of a breach. We formalize and study the corresponding problem of finding generalizations that both maximize data utility and minimize empirical privacy risk, which we quantify by introducing a diverse set of policy-aligned adversarial scenarios. Finally, we propose a range of baseline vDM algorithms, as well as Privacy-aware Tree (PAT), an especially effective vDM algorithm that outperforms all baselines across several settings. We plan to release our code as a publicly available library, helping advance the standardization of DM for machine learning. Overall, we believe our work can help lay the foundation for further exploration and adoption of DM principles in real-world applications.



## **4. Hard Label Black Box Node Injection Attack on Graph Neural Networks**

cs.LG

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13244v1) [paper-pdf](http://arxiv.org/pdf/2311.13244v1)

**Authors**: Yu Zhou, Zihao Dong, Guofeng Zhang, Jingchen Tang

**Abstract**: While graph neural networks have achieved state-of-the-art performances in many real-world tasks including graph classification and node classification, recent works have demonstrated they are also extremely vulnerable to adversarial attacks. Most previous works have focused on attacking node classification networks under impractical white-box scenarios. In this work, we will propose a non-targeted Hard Label Black Box Node Injection Attack on Graph Neural Networks, which to the best of our knowledge, is the first of its kind. Under this setting, more real world tasks can be studied because our attack assumes no prior knowledge about (1): the model architecture of the GNN we are attacking; (2): the model's gradients; (3): the output logits of the target GNN model. Our attack is based on an existing edge perturbation attack, from which we restrict the optimization process to formulate a node injection attack. In the work, we will evaluate the performance of the attack using three datasets, COIL-DEL, IMDB-BINARY, and NCI1.



## **5. A Survey of Adversarial CAPTCHAs on its History, Classification and Generation**

cs.CR

Submitted to ACM Computing Surveys (Under Review)

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13233v1) [paper-pdf](http://arxiv.org/pdf/2311.13233v1)

**Authors**: Zisheng Xu, Qiao Yan, F. Richard Yu, Victor C. M. Leung

**Abstract**: Completely Automated Public Turing test to tell Computers and Humans Apart, short for CAPTCHA, is an essential and relatively easy way to defend against malicious attacks implemented by bots. The security and usability trade-off limits the use of massive geometric transformations to interfere deep model recognition and deep models even outperformed humans in complex CAPTCHAs. The discovery of adversarial examples provides an ideal solution to the security and usability trade-off by integrating adversarial examples and CAPTCHAs to generate adversarial CAPTCHAs that can fool the deep models. In this paper, we extend the definition of adversarial CAPTCHAs and propose a classification method for adversarial CAPTCHAs. Then we systematically review some commonly used methods to generate adversarial examples and methods that are successfully used to generate adversarial CAPTCHAs. Also, we analyze some defense methods that can be used to defend adversarial CAPTCHAs, indicating potential threats to adversarial CAPTCHAs. Finally, we discuss some possible future research directions for adversarial CAPTCHAs at the end of this paper.



## **6. HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks**

cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2309.08549v3) [paper-pdf](http://arxiv.org/pdf/2309.08549v3)

**Authors**: Minh-Hao Van, Alycia N. Carey, Xintao Wu

**Abstract**: While numerous defense methods have been proposed to prohibit potential poisoning attacks from untrusted data sources, most research works only defend against specific attacks, which leaves many avenues for an adversary to exploit. In this work, we propose an efficient and robust training approach to defend against data poisoning attacks based on influence functions, named Healthy Influential-Noise based Training. Using influence functions, we craft healthy noise that helps to harden the classification model against poisoning attacks without significantly affecting the generalization ability on test data. In addition, our method can perform effectively when only a subset of the training data is modified, instead of the current method of adding noise to all examples that has been used in several previous works. We conduct comprehensive evaluations over two image datasets with state-of-the-art poisoning attacks under different realistic attack scenarios. Our empirical results show that HINT can efficiently protect deep learning models against the effect of both untargeted and targeted poisoning attacks.



## **7. Epsilon*: Privacy Metric for Machine Learning Models**

cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2307.11280v2) [paper-pdf](http://arxiv.org/pdf/2307.11280v2)

**Authors**: Diana M. Negoescu, Humberto Gonzalez, Saad Eddin Al Orjany, Jilei Yang, Yuliia Lut, Rahul Tandra, Xiaowen Zhang, Xinyi Zheng, Zach Douglas, Vidita Nolkha, Parvez Ahammad, Gennady Samorodnitsky

**Abstract**: We introduce Epsilon*, a new privacy metric for measuring the privacy risk of a single model instance prior to, during, or after deployment of privacy mitigation strategies. The metric requires only black-box access to model predictions, does not require training data re-sampling or model re-training, and can be used to measure the privacy risk of models not trained with differential privacy. Epsilon* is a function of true positive and false positive rates in a hypothesis test used by an adversary in a membership inference attack. We distinguish between quantifying the privacy loss of a trained model instance, which we refer to as empirical privacy, and quantifying the privacy loss of the training mechanism which produces this model instance. Existing approaches in the privacy auditing literature provide lower bounds for the latter, while our metric provides an empirical lower bound for the former by relying on an (${\epsilon}$, ${\delta}$)-type of quantification of the privacy of the trained model instance. We establish a relationship between these lower bounds and show how to implement Epsilon* to avoid numerical and noise amplification instability. We further show in experiments on benchmark public data sets that Epsilon* is sensitive to privacy risk mitigation by training with differential privacy (DP), where the value of Epsilon* is reduced by up to 800% compared to the Epsilon* values of non-DP trained baseline models. This metric allows privacy auditors to be independent of model owners, and enables visualizing the privacy-utility landscape to make informed decisions regarding the trade-offs between model privacy and utility.



## **8. Is your vote truly secret? Ballot Secrecy iff Ballot Independence: Proving necessary conditions and analysing case studies**

cs.CR

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12977v1) [paper-pdf](http://arxiv.org/pdf/2311.12977v1)

**Authors**: Aida Manzano Kharman, Ben Smyth, Freddie Page

**Abstract**: We formalise definitions of ballot secrecy and ballot independence by Smyth, JCS'21 as indistinguishability games in the computational model of security. These definitions improve upon Smyth, draft '21 to consider a wider class of voting systems. Both Smyth, JCS'21 and Smyth, draft '21 improve on earlier works by considering a more realistic adversary model wherein they have access to the ballot collection. We prove that ballot secrecy implies ballot independence. We say ballot independence holds if a system has non-malleable ballots. We construct games for ballot secrecy and non-malleability and show that voting schemes with malleable ballots do not preserve ballot secrecy. We demonstrate that Helios does not satisfy our definition of ballot secrecy. Furthermore, the Python framework we constructed for our case study shows that if an attack exists against non-malleability, this attack can be used to break ballot secrecy.



## **9. Iris Presentation Attack: Assessing the Impact of Combining Vanadium Dioxide Films with Artificial Eyes**

cs.CV

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12773v1) [paper-pdf](http://arxiv.org/pdf/2311.12773v1)

**Authors**: Darshika Jauhari, Renu Sharma, Cunjian Chen, Nelson Sepulveda, Arun Ross

**Abstract**: Iris recognition systems, operating in the near infrared spectrum (NIR), have demonstrated vulnerability to presentation attacks, where an adversary uses artifacts such as cosmetic contact lenses, artificial eyes or printed iris images in order to circumvent the system. At the same time, a number of effective presentation attack detection (PAD) methods have been developed. These methods have demonstrated success in detecting artificial eyes (e.g., fake Van Dyke eyes) as presentation attacks. In this work, we seek to alter the optical characteristics of artificial eyes by affixing Vanadium Dioxide (VO2) films on their surface in various spatial configurations. VO2 films can be used to selectively transmit NIR light and can, therefore, be used to regulate the amount of NIR light from the object that is captured by the iris sensor. We study the impact of such images produced by the sensor on two state-of-the-art iris PA detection methods. We observe that the addition of VO2 films on the surface of artificial eyes can cause the PA detection methods to misclassify them as bonafide eyes in some cases. This represents a vulnerability that must be systematically analyzed and effectively addressed.



## **10. Attention Deficit is Ordered! Fooling Deformable Vision Transformers with Collaborative Adversarial Patches**

cs.CV

9 pages, 10 figures

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12914v1) [paper-pdf](http://arxiv.org/pdf/2311.12914v1)

**Authors**: Quazi Mishkatul Alam, Bilel Tarchoun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: The latest generation of transformer-based vision models have proven to be superior to Convolutional Neural Network (CNN)-based models across several vision tasks, largely attributed to their remarkable prowess in relation modeling. Deformable vision transformers significantly reduce the quadratic complexity of modeling attention by using sparse attention structures, enabling them to be used in larger scale applications such as multi-view vision systems. Recent work demonstrated adversarial attacks against transformers; we show that these attacks do not transfer to deformable transformers due to their sparse attention structure. Specifically, attention in deformable transformers is modeled using pointers to the most relevant other tokens. In this work, we contribute for the first time adversarial attacks that manipulate the attention of deformable transformers, distracting them to focus on irrelevant parts of the image. We also develop new collaborative attacks where a source patch manipulates attention to point to a target patch that adversarially attacks the system. In our experiments, we find that only 1% patched area of the input field can lead to 0% AP. We also show that the attacks provide substantial versatility to support different attacker scenarios because of their ability to redirect attention under the attacker control.



## **11. BrainWash: A Poisoning Attack to Forget in Continual Learning**

cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.11995v2) [paper-pdf](http://arxiv.org/pdf/2311.11995v2)

**Authors**: Ali Abbasi, Parsa Nooralinejad, Hamed Pirsiavash, Soheil Kolouri

**Abstract**: Continual learning has gained substantial attention within the deep learning community, offering promising solutions to the challenging problem of sequential learning. Yet, a largely unexplored facet of this paradigm is its susceptibility to adversarial attacks, especially with the aim of inducing forgetting. In this paper, we introduce "BrainWash," a novel data poisoning method tailored to impose forgetting on a continual learner. By adding the BrainWash noise to a variety of baselines, we demonstrate how a trained continual learner can be induced to forget its previously learned tasks catastrophically, even when using these continual learning baselines. An important feature of our approach is that the attacker requires no access to previous tasks' data and is armed merely with the model's current parameters and the data belonging to the most recent task. Our extensive experiments highlight the efficacy of BrainWash, showcasing degradation in performance across various regularization-based continual learning methods.



## **12. Attacking Motion Planners Using Adversarial Perception Errors**

cs.RO

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12722v1) [paper-pdf](http://arxiv.org/pdf/2311.12722v1)

**Authors**: Jonathan Sadeghi, Nicholas A. Lord, John Redford, Romain Mueller

**Abstract**: Autonomous driving (AD) systems are often built and tested in a modular fashion, where the performance of different modules is measured using task-specific metrics. These metrics should be chosen so as to capture the downstream impact of each module and the performance of the system as a whole. For example, high perception quality should enable prediction and planning to be performed safely. Even though this is true in general, we show here that it is possible to construct planner inputs that score very highly on various perception quality metrics but still lead to planning failures. In an analogy to adversarial attacks on image classifiers, we call such inputs \textbf{adversarial perception errors} and show they can be systematically constructed using a simple boundary-attack algorithm. We demonstrate the effectiveness of this algorithm by finding attacks for two different black-box planners in several urban and highway driving scenarios using the CARLA simulator. Finally, we analyse the properties of these attacks and show that they are isolated in the input space of the planner, and discuss their implications for AD system deployment and testing.



## **13. Differentially Private Optimizers Can Learn Adversarially Robust Models**

cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2211.08942v2) [paper-pdf](http://arxiv.org/pdf/2211.08942v2)

**Authors**: Yuan Zhang, Zhiqi Bu

**Abstract**: Machine learning models have shone in a variety of domains and attracted increasing attention from both the security and the privacy communities. One important yet worrying question is: Will training models under the differential privacy (DP) constraint have an unfavorable impact on their adversarial robustness? While previous works have postulated that privacy comes at the cost of worse robustness, we give the first theoretical analysis to show that DP models can indeed be robust and accurate, even sometimes more robust than their naturally-trained non-private counterparts. We observe three key factors that influence the privacy-robustness-accuracy tradeoff: (1) hyper-parameters for DP optimizers are critical; (2) pre-training on public data significantly mitigates the accuracy and robustness drop; (3) choice of DP optimizers makes a difference. With these factors set properly, we achieve 90\% natural accuracy, 72\% robust accuracy ($+9\%$ than the non-private model) under $l_2(0.5)$ attack, and 69\% robust accuracy ($+16\%$ than the non-private model) with pre-trained SimCLRv2 model under $l_\infty(4/255)$ attack on CIFAR10 with $\epsilon=2$. In fact, we show both theoretically and empirically that DP models are Pareto optimal on the accuracy-robustness tradeoff. Empirically, the robustness of DP models is consistently observed across various datasets and models. We believe our encouraging results are a significant step towards training models that are private as well as robust.



## **14. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

cs.CL

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2309.01446v3) [paper-pdf](http://arxiv.org/pdf/2309.01446v3)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.



## **15. Beyond Labeling Oracles: What does it mean to steal ML models?**

cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2310.01959v2) [paper-pdf](http://arxiv.org/pdf/2310.01959v2)

**Authors**: Avital Shafran, Ilia Shumailov, Murat A. Erdogdu, Nicolas Papernot

**Abstract**: Model extraction attacks are designed to steal trained models with only query access, as is often provided through APIs that ML-as-a-Service providers offer. ML models are expensive to train, in part because data is hard to obtain, and a primary incentive for model extraction is to acquire a model while incurring less cost than training from scratch. Literature on model extraction commonly claims or presumes that the attacker is able to save on both data acquisition and labeling costs. We show that the attacker often does not. This is because current attacks implicitly rely on the adversary being able to sample from the victim model's data distribution. We thoroughly evaluate factors influencing the success of model extraction. We discover that prior knowledge of the attacker, i.e. access to in-distribution data, dominates other factors like the attack policy the adversary follows to choose which queries to make to the victim model API. Thus, an adversary looking to develop an equally capable model with a fixed budget has little practical incentive to perform model extraction, since for the attack to work they need to collect in-distribution data, saving only on the cost of labeling. With low labeling costs in the current market, the usefulness of such attacks is questionable. Ultimately, we demonstrate that the effect of prior knowledge needs to be explicitly decoupled from the attack policy. To this end, we propose a benchmark to evaluate attack policy directly.



## **16. Malicious URL Detection via Pretrained Language Model Guided Multi-Level Feature Attention Network**

cs.CR

11 pages, 7 figures

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12372v1) [paper-pdf](http://arxiv.org/pdf/2311.12372v1)

**Authors**: Ruitong Liu, Yanbin Wang, Haitao Xu, Zhan Qin, Yiwei Liu, Zheng Cao

**Abstract**: The widespread use of the Internet has revolutionized information retrieval methods. However, this transformation has also given rise to a significant cybersecurity challenge: the rapid proliferation of malicious URLs, which serve as entry points for a wide range of cyber threats. In this study, we present an efficient pre-training model-based framework for malicious URL detection. Leveraging the subword and character-aware pre-trained model, CharBERT, as our foundation, we further develop three key modules: hierarchical feature extraction, layer-aware attention, and spatial pyramid pooling. The hierarchical feature extraction module follows the pyramid feature learning principle, extracting multi-level URL embeddings from the different Transformer layers of CharBERT. Subsequently, the layer-aware attention module autonomously learns connections among features at various hierarchical levels and allocates varying weight coefficients to each level of features. Finally, the spatial pyramid pooling module performs multiscale downsampling on the weighted multi-level feature pyramid, achieving the capture of local features as well as the aggregation of global features. The proposed method has been extensively validated on multiple public datasets, demonstrating a significant improvement over prior works, with the maximum accuracy gap reaching 8.43% compared to the previous state-of-the-art method. Additionally, we have assessed the model's generalization and robustness in scenarios such as cross-dataset evaluation and adversarial attacks. Finally, we conducted real-world case studies on the active phishing URLs.



## **17. Rethinking the Backward Propagation for Adversarial Transferability**

cs.CV

Accepted by NeurIPS 2023

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2306.12685v3) [paper-pdf](http://arxiv.org/pdf/2306.12685v3)

**Authors**: Xiaosen Wang, Kangheng Tong, Kun He

**Abstract**: Transfer-based attacks generate adversarial examples on the surrogate model, which can mislead other black-box models without access, making it promising to attack real-world applications. Recently, several works have been proposed to boost adversarial transferability, in which the surrogate model is usually overlooked. In this work, we identify that non-linear layers (e.g., ReLU, max-pooling, etc.) truncate the gradient during backward propagation, making the gradient w.r.t. input image imprecise to the loss function. We hypothesize and empirically validate that such truncation undermines the transferability of adversarial examples. Based on these findings, we propose a novel method called Backward Propagation Attack (BPA) to increase the relevance between the gradient w.r.t. input image and loss function so as to generate adversarial examples with higher transferability. Specifically, BPA adopts a non-monotonic function as the derivative of ReLU and incorporates softmax with temperature to smooth the derivative of max-pooling, thereby mitigating the information loss during the backward propagation of gradients. Empirical results on the ImageNet dataset demonstrate that not only does our method substantially boost the adversarial transferability, but it is also general to existing transfer-based attacks. Code is available at https://github.com/Trustworthy-AI-Group/RPA.



## **18. Resilient Control of Networked Microgrids using Vertical Federated Reinforcement Learning: Designs and Real-Time Test-Bed Validations**

eess.SY

10 pages, 7 figures

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12264v1) [paper-pdf](http://arxiv.org/pdf/2311.12264v1)

**Authors**: Sayak Mukherjee, Ramij R. Hossain, Sheik M. Mohiuddin, Yuan Liu, Wei Du, Veronica Adetola, Rohit A. Jinsiwale, Qiuhua Huang, Tianzhixi Yin, Ankit Singhal

**Abstract**: Improving system-level resiliency of networked microgrids is an important aspect with increased population of inverter-based resources (IBRs). This paper (1) presents resilient control design in presence of adversarial cyber-events, and proposes a novel federated reinforcement learning (Fed-RL) approach to tackle (a) model complexities, unknown dynamical behaviors of IBR devices, (b) privacy issues regarding data sharing in multi-party-owned networked grids, and (2) transfers learned controls from simulation to hardware-in-the-loop test-bed, thereby bridging the gap between simulation and real world. With these multi-prong objectives, first, we formulate a reinforcement learning (RL) training setup generating episodic trajectories with adversaries (attack signal) injected at the primary controllers of the grid forming (GFM) inverters where RL agents (or controllers) are being trained to mitigate the injected attacks. For networked microgrids, the horizontal Fed-RL method involving distinct independent environments is not appropriate, leading us to develop vertical variant Federated Soft Actor-Critic (FedSAC) algorithm to grasp the interconnected dynamics of networked microgrid. Next, utilizing OpenAI Gym interface, we built a custom simulation set-up in GridLAB-D/HELICS co-simulation platform, named Resilient RL Co-simulation (ResRLCoSIM), to train the RL agents with IEEE 123-bus benchmark test systems comprising 3 interconnected microgrids. Finally, the learned policies in simulation world are transferred to the real-time hardware-in-the-loop test-bed set-up developed using high-fidelity Hypersim platform. Experiments show that the simulator-trained RL controllers produce convincing results with the real-time test-bed set-up, validating the minimization of sim-to-real gap.



## **19. DefensiveDR: Defending against Adversarial Patches using Dimensionality Reduction**

cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.12211v1) [paper-pdf](http://arxiv.org/pdf/2311.12211v1)

**Authors**: Nandish Chattopadhyay, Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique

**Abstract**: Adversarial patch-based attacks have shown to be a major deterrent towards the reliable use of machine learning models. These attacks involve the strategic modification of localized patches or specific image areas to deceive trained machine learning models. In this paper, we propose \textit{DefensiveDR}, a practical mechanism using a dimensionality reduction technique to thwart such patch-based attacks. Our method involves projecting the sample images onto a lower-dimensional space while retaining essential information or variability for effective machine learning tasks. We perform this using two techniques, Singular Value Decomposition and t-Distributed Stochastic Neighbor Embedding. We experimentally tune the variability to be preserved for optimal performance as a hyper-parameter. This dimension reduction substantially mitigates adversarial perturbations, thereby enhancing the robustness of the given machine learning model. Our defense is model-agnostic and operates without assumptions about access to model decisions or model architectures, making it effective in both black-box and white-box settings. Furthermore, it maintains accuracy across various models and remains robust against several unseen patch-based attacks. The proposed defensive approach improves the accuracy from 38.8\% (without defense) to 66.2\% (with defense) when performing LaVAN and GoogleAp attacks, which supersedes that of the prominent state-of-the-art like LGS (53.86\%) and Jujutsu (60\%).



## **20. Generating Valid and Natural Adversarial Examples with Large Language Models**

cs.CL

Submitted to the IEEE for possible publication

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11861v1) [paper-pdf](http://arxiv.org/pdf/2311.11861v1)

**Authors**: Zimu Wang, Wei Wang, Qi Chen, Qiufeng Wang, Anh Nguyen

**Abstract**: Deep learning-based natural language processing (NLP) models, particularly pre-trained language models (PLMs), have been revealed to be vulnerable to adversarial attacks. However, the adversarial examples generated by many mainstream word-level adversarial attack models are neither valid nor natural, leading to the loss of semantic maintenance, grammaticality, and human imperceptibility. Based on the exceptional capacity of language understanding and generation of large language models (LLMs), we propose LLM-Attack, which aims at generating both valid and natural adversarial examples with LLMs. The method consists of two stages: word importance ranking (which searches for the most vulnerable words) and word synonym replacement (which substitutes them with their synonyms obtained from LLMs). Experimental results on the Movie Review (MR), IMDB, and Yelp Review Polarity datasets against the baseline adversarial attack models illustrate the effectiveness of LLM-Attack, and it outperforms the baselines in human and GPT-4 evaluation by a significant margin. The model can generate adversarial examples that are typically valid and natural, with the preservation of semantic meaning, grammaticality, and human imperceptibility.



## **21. Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems**

cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11796v1) [paper-pdf](http://arxiv.org/pdf/2311.11796v1)

**Authors**: Guangjing Wang, Ce Zhou, Yuanda Wang, Bocheng Chen, Hanqing Guo, Qiben Yan

**Abstract**: Artificial Intelligence (AI) systems such as autonomous vehicles, facial recognition, and speech recognition systems are increasingly integrated into our daily lives. However, despite their utility, these AI systems are vulnerable to a wide range of attacks such as adversarial, backdoor, data poisoning, membership inference, model inversion, and model stealing attacks. In particular, numerous attacks are designed to target a particular model or system, yet their effects can spread to additional targets, referred to as transferable attacks. Although considerable efforts have been directed toward developing transferable attacks, a holistic understanding of the advancements in transferable attacks remains elusive. In this paper, we comprehensively explore learning-based attacks from the perspective of transferability, particularly within the context of cyber-physical security. We delve into different domains -- the image, text, graph, audio, and video domains -- to highlight the ubiquitous and pervasive nature of transferable attacks. This paper categorizes and reviews the architecture of existing attacks from various viewpoints: data, process, model, and system. We further examine the implications of transferable attacks in practical scenarios such as autonomous driving, speech recognition, and large language models (LLMs). Additionally, we outline the potential research directions to encourage efforts in exploring the landscape of transferable attacks. This survey offers a holistic understanding of the prevailing transferable attacks and their impacts across different domains.



## **22. AdvGen: Physical Adversarial Attack on Face Presentation Attack Detection Systems**

cs.CV

10 pages, 9 figures, Accepted to the International Joint Conference  on Biometrics (IJCB 2023)

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11753v1) [paper-pdf](http://arxiv.org/pdf/2311.11753v1)

**Authors**: Sai Amrit Patnaik, Shivali Chansoriya, Anil K. Jain, Anoop M. Namboodiri

**Abstract**: Evaluating the risk level of adversarial images is essential for safely deploying face authentication models in the real world. Popular approaches for physical-world attacks, such as print or replay attacks, suffer from some limitations, like including physical and geometrical artifacts. Recently, adversarial attacks have gained attraction, which try to digitally deceive the learning strategy of a recognition system using slight modifications to the captured image. While most previous research assumes that the adversarial image could be digitally fed into the authentication systems, this is not always the case for systems deployed in the real world. This paper demonstrates the vulnerability of face authentication systems to adversarial images in physical world scenarios. We propose AdvGen, an automated Generative Adversarial Network, to simulate print and replay attacks and generate adversarial images that can fool state-of-the-art PADs in a physical domain attack setting. Using this attack strategy, the attack success rate goes up to 82.01%. We test AdvGen extensively on four datasets and ten state-of-the-art PADs. We also demonstrate the effectiveness of our attack by conducting experiments in a realistic, physical environment.



## **23. APARATE: Adaptive Adversarial Patch for CNN-based Monocular Depth Estimation for Autonomous Navigation**

cs.CV

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2303.01351v2) [paper-pdf](http://arxiv.org/pdf/2303.01351v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Ihsen Alouani, Muhammad Shafique

**Abstract**: In recent times, monocular depth estimation (MDE) has experienced significant advancements in performance, largely attributed to the integration of innovative architectures, i.e., convolutional neural networks (CNNs) and Transformers. Nevertheless, the susceptibility of these models to adversarial attacks has emerged as a noteworthy concern, especially in domains where safety and security are paramount. This concern holds particular weight for MDE due to its critical role in applications like autonomous driving and robotic navigation, where accurate scene understanding is pivotal. To assess the vulnerability of CNN-based depth prediction methods, recent work tries to design adversarial patches against MDE. However, the existing approaches fall short of inducing a comprehensive and substantially disruptive impact on the vision system. Instead, their influence is partial and confined to specific local areas. These methods lead to erroneous depth predictions only within the overlapping region with the input image, without considering the characteristics of the target object, such as its size, shape, and position. In this paper, we introduce a novel adversarial patch named APARATE. This patch possesses the ability to selectively undermine MDE in two distinct ways: by distorting the estimated distances or by creating the illusion of an object disappearing from the perspective of the autonomous system. Notably, APARATE is designed to be sensitive to the shape and scale of the target object, and its influence extends beyond immediate proximity. APARATE, results in a mean depth estimation error surpassing $0.5$, significantly impacting as much as $99\%$ of the targeted region when applied to CNN-based MDE models. Furthermore, it yields a significant error of $0.34$ and exerts substantial influence over $94\%$ of the target region in the context of Transformer-based MDE.



## **24. DAP: A Dynamic Adversarial Patch for Evading Person Detectors**

cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2305.11618v2) [paper-pdf](http://arxiv.org/pdf/2305.11618v2)

**Authors**: Amira Guesmi, Ruitian Ding, Muhammad Abdullah Hanif, Ihsen Alouani, Muhammad Shafique

**Abstract**: Patch-based adversarial attacks were proven to compromise the robustness and reliability of computer vision systems. However, their conspicuous and easily detectable nature challenge their practicality in real-world setting. To address this, recent work has proposed using Generative Adversarial Networks (GANs) to generate naturalistic patches that may not attract human attention. However, such approaches suffer from a limited latent space making it challenging to produce a patch that is efficient, stealthy, and robust to multiple real-world transformations. This paper introduces a novel approach that produces a Dynamic Adversarial Patch (DAP) designed to overcome these limitations. DAP maintains a naturalistic appearance while optimizing attack efficiency and robustness to real-world transformations. The approach involves redefining the optimization problem and introducing a novel objective function that incorporates a similarity metric to guide the patch's creation. Unlike GAN-based techniques, the DAP directly modifies pixel values within the patch, providing increased flexibility and adaptability to multiple transformations. Furthermore, most clothing-based physical attacks assume static objects and ignore the possible transformations caused by non-rigid deformation due to changes in a person's pose. To address this limitation, a 'Creases Transformation' (CT) block is introduced, enhancing the patch's resilience to a variety of real-world distortions. Experimental results demonstrate that the proposed approach outperforms state-of-the-art attacks, achieving a success rate of up to 82.28% in the digital world when targeting the YOLOv7 detector and 65% in the physical world when targeting YOLOv3tiny detector deployed in edge-based smart cameras.



## **25. ODDR: Outlier Detection & Dimension Reduction Based Defense Against Adversarial Patches**

cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.12084v1) [paper-pdf](http://arxiv.org/pdf/2311.12084v1)

**Authors**: Nandish Chattopadhyay, Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique

**Abstract**: Adversarial attacks are a major deterrent towards the reliable use of machine learning models. A powerful type of adversarial attacks is the patch-based attack, wherein the adversarial perturbations modify localized patches or specific areas within the images to deceive the trained machine learning model. In this paper, we introduce Outlier Detection and Dimension Reduction (ODDR), a holistic defense mechanism designed to effectively mitigate patch-based adversarial attacks. In our approach, we posit that input features corresponding to adversarial patches, whether naturalistic or otherwise, deviate from the inherent distribution of the remaining image sample and can be identified as outliers or anomalies. ODDR employs a three-stage pipeline: Fragmentation, Segregation, and Neutralization, providing a model-agnostic solution applicable to both image classification and object detection tasks. The Fragmentation stage parses the samples into chunks for the subsequent Segregation process. Here, outlier detection techniques identify and segregate the anomalous features associated with adversarial perturbations. The Neutralization stage utilizes dimension reduction methods on the outliers to mitigate the impact of adversarial perturbations without sacrificing pertinent information necessary for the machine learning task. Extensive testing on benchmark datasets and state-of-the-art adversarial patches demonstrates the effectiveness of ODDR. Results indicate robust accuracies matching and lying within a small range of clean accuracies (1%-3% for classification and 3%-5% for object detection), with only a marginal compromise of 1%-2% in performance on clean samples, thereby significantly outperforming other defenses.



## **26. Understanding Variation in Subpopulation Susceptibility to Poisoning Attacks**

cs.LG

18 pages, 11 figures

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11544v1) [paper-pdf](http://arxiv.org/pdf/2311.11544v1)

**Authors**: Evan Rose, Fnu Suya, David Evans

**Abstract**: Machine learning is susceptible to poisoning attacks, in which an attacker controls a small fraction of the training data and chooses that data with the goal of inducing some behavior unintended by the model developer in the trained model. We consider a realistic setting in which the adversary with the ability to insert a limited number of data points attempts to control the model's behavior on a specific subpopulation. Inspired by previous observations on disparate effectiveness of random label-flipping attacks on different subpopulations, we investigate the properties that can impact the effectiveness of state-of-the-art poisoning attacks against different subpopulations. For a family of 2-dimensional synthetic datasets, we empirically find that dataset separability plays a dominant role in subpopulation vulnerability for less separable datasets. However, well-separated datasets exhibit more dependence on individual subpopulation properties. We further discover that a crucial subpopulation property is captured by the difference in loss on the clean dataset between the clean model and a target model that misclassifies the subpopulation, and a subpopulation is much easier to attack if the loss difference is small. This property also generalizes to high-dimensional benchmark datasets. For the Adult benchmark dataset, we show that we can find semantically-meaningful subpopulation properties that are related to the susceptibilities of a selected group of subpopulations. The results in this paper are accompanied by a fully interactive web-based visualization of subpopulation poisoning attacks found at https://uvasrg.github.io/visualizing-poisoning



## **27. Assessing Prompt Injection Risks in 200+ Custom GPTs**

cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11538v1) [paper-pdf](http://arxiv.org/pdf/2311.11538v1)

**Authors**: Jiahao Yu, Yuhang Wu, Dong Shu, Mingyu Jin, Xinyu Xing

**Abstract**: In the rapidly evolving landscape of artificial intelligence, ChatGPT has been widely used in various applications. The new feature: customization of ChatGPT models by users to cater to specific needs has opened new frontiers in AI utility. However, this study reveals a significant security vulnerability inherent in these user-customized GPTs: prompt injection attacks. Through comprehensive testing of over 200 user-designed GPT models via adversarial prompts, we demonstrate that these systems are susceptible to prompt injections. Through prompt injection, an adversary can not only extract the customized system prompts but also access the uploaded files. This paper provides a first-hand analysis of the prompt injection, alongside the evaluation of the possible mitigation of such attacks. Our findings underscore the urgent need for robust security frameworks in the design and deployment of customizable GPT models. The intent of this paper is to raise awareness and prompt action in the AI community, ensuring that the benefits of GPT customization do not come at the cost of compromised security and privacy.



## **28. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

cs.CL

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11509v1) [paper-pdf](http://arxiv.org/pdf/2311.11509v1)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Vishy Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.



## **29. Interpretable Computer Vision Models through Adversarial Training: Unveiling the Robustness-Interpretability Connection**

cs.CV

13 pages, 19 figures, 6 tables

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2307.02500v2) [paper-pdf](http://arxiv.org/pdf/2307.02500v2)

**Authors**: Delyan Boychev

**Abstract**: With the perpetual increase of complexity of the state-of-the-art deep neural networks, it becomes a more and more challenging task to maintain their interpretability. Our work aims to evaluate the effects of adversarial training utilized to produce robust models - less vulnerable to adversarial attacks. It has been shown to make computer vision models more interpretable. Interpretability is as essential as robustness when we deploy the models to the real world. To prove the correlation between these two problems, we extensively examine the models using local feature-importance methods (SHAP, Integrated Gradients) and feature visualization techniques (Representation Inversion, Class Specific Image Generation). Standard models, compared to robust are more susceptible to adversarial attacks, and their learned representations are less meaningful to humans. Conversely, these models focus on distinctive regions of the images which support their predictions. Moreover, the features learned by the robust model are closer to the real ones.



## **30. Revisiting and Advancing Adversarial Training Through A Simple Baseline**

cs.CV

11 pages, 8 figures

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2306.07613v2) [paper-pdf](http://arxiv.org/pdf/2306.07613v2)

**Authors**: Hong Liu

**Abstract**: In this paper, we delve into the essential components of adversarial training which is a pioneering defense technique against adversarial attacks. We indicate that some factors such as the loss function, learning rate scheduler, and data augmentation, which are independent of the model architecture, will influence adversarial robustness and generalization. When these factors are controlled for, we introduce a simple baseline approach, termed SimpleAT, that performs competitively with recent methods and mitigates robust overfitting. We conduct extensive experiments on CIFAR-10/100 and Tiny-ImageNet, which validate the robustness of SimpleAT against state-of-the-art adversarial attackers such as AutoAttack. Our results also demonstrate that SimpleAT exhibits good performance in the presence of various image corruptions, such as those found in the CIFAR-10-C. In addition, we empirically show that SimpleAT is capable of reducing the variance in model predictions, which is considered the primary contributor to robust overfitting. Our results also reveal the connections between SimpleAT and many advanced state-of-the-art adversarial defense methods.



## **31. Robust Network Pruning With Sparse Entropic Wasserstein Regression**

cs.AI

submitted to ICLR 2024

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2310.04918v2) [paper-pdf](http://arxiv.org/pdf/2310.04918v2)

**Authors**: Lei You, Hei Victor Cheng

**Abstract**: This study tackles the issue of neural network pruning that inaccurate gradients exist when computing the empirical Fisher Information Matrix (FIM). We introduce an entropic Wasserstein regression (EWR) formulation, capitalizing on the geometric attributes of the optimal transport (OT) problem. This is analytically showcased to excel in noise mitigation by adopting neighborhood interpolation across data points. The unique strength of the Wasserstein distance is its intrinsic ability to strike a balance between noise reduction and covariance information preservation. Extensive experiments performed on various networks show comparable performance of the proposed method with state-of-the-art (SoTA) network pruning algorithms. Our proposed method outperforms the SoTA when the network size or the target sparsity is large, the gain is even larger with the existence of noisy gradients, possibly from noisy data, analog memory, or adversarial attacks. Notably, our proposed method achieves a gain of 6% improvement in accuracy and 8% improvement in testing loss for MobileNetV1 with less than one-fourth of the network parameters remaining.



## **32. Adversarial Prompt Tuning for Vision-Language Models**

cs.CV

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.11261v1) [paper-pdf](http://arxiv.org/pdf/2311.11261v1)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code will be available upon publication of the paper.



## **33. Untargeted Black-box Attacks for Social Recommendations**

cs.SI

Preprint. Under review

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.07127v2) [paper-pdf](http://arxiv.org/pdf/2311.07127v2)

**Authors**: Wenqi Fan, Shijie Wang, Xiao-yong Wei, Xiaowei Mei, Qing Li

**Abstract**: The rise of online social networks has facilitated the evolution of social recommender systems, which incorporate social relations to enhance users' decision-making process. With the great success of Graph Neural Networks in learning node representations, GNN-based social recommendations have been widely studied to model user-item interactions and user-user social relations simultaneously. Despite their great successes, recent studies have shown that these advanced recommender systems are highly vulnerable to adversarial attacks, in which attackers can inject well-designed fake user profiles to disrupt recommendation performances. While most existing studies mainly focus on targeted attacks to promote target items on vanilla recommender systems, untargeted attacks to degrade the overall prediction performance are less explored on social recommendations under a black-box scenario. To perform untargeted attacks on social recommender systems, attackers can construct malicious social relationships for fake users to enhance the attack performance. However, the coordination of social relations and item profiles is challenging for attacking black-box social recommendations. To address this limitation, we first conduct several preliminary studies to demonstrate the effectiveness of cross-community connections and cold-start items in degrading recommendations performance. Specifically, we propose a novel framework Multiattack based on multi-agent reinforcement learning to coordinate the generation of cold-start item profiles and cross-community social relations for conducting untargeted attacks on black-box social recommendations. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of our proposed attacking framework under the black-box setting.



## **34. Robust Network Slicing: Multi-Agent Policies, Adversarial Attacks, and Defensive Strategies**

cs.LG

Published in IEEE Transactions on Machine Learning in Communications  and Networking (TMLCN)

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.11206v1) [paper-pdf](http://arxiv.org/pdf/2311.11206v1)

**Authors**: Feng Wang, M. Cenk Gursoy, Senem Velipasalar

**Abstract**: In this paper, we present a multi-agent deep reinforcement learning (deep RL) framework for network slicing in a dynamic environment with multiple base stations and multiple users. In particular, we propose a novel deep RL framework with multiple actors and centralized critic (MACC) in which actors are implemented as pointer networks to fit the varying dimension of input. We evaluate the performance of the proposed deep RL algorithm via simulations to demonstrate its effectiveness. Subsequently, we develop a deep RL based jammer with limited prior information and limited power budget. The goal of the jammer is to minimize the transmission rates achieved with network slicing and thus degrade the network slicing agents' performance. We design a jammer with both listening and jamming phases and address jamming location optimization as well as jamming channel optimization via deep RL. We evaluate the jammer at the optimized location, generating interference attacks in the optimized set of channels by switching between the jamming phase and listening phase. We show that the proposed jammer can significantly reduce the victims' performance without direct feedback or prior knowledge on the network slicing policies. Finally, we devise a Nash-equilibrium-supervised policy ensemble mixed strategy profile for network slicing (as a defensive measure) and jamming. We evaluate the performance of the proposed policy ensemble algorithm by applying on the network slicing agents and the jammer agent in simulations to show its effectiveness.



## **35. Attention-Based Real-Time Defenses for Physical Adversarial Attacks in Vision Applications**

cs.CV

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.11191v1) [paper-pdf](http://arxiv.org/pdf/2311.11191v1)

**Authors**: Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: Deep neural networks exhibit excellent performance in computer vision tasks, but their vulnerability to real-world adversarial attacks, achieved through physical objects that can corrupt their predictions, raises serious security concerns for their application in safety-critical domains. Existing defense methods focus on single-frame analysis and are characterized by high computational costs that limit their applicability in multi-frame scenarios, where real-time decisions are crucial.   To address this problem, this paper proposes an efficient attention-based defense mechanism that exploits adversarial channel-attention to quickly identify and track malicious objects in shallow network layers and mask their adversarial effects in a multi-frame setting. This work advances the state of the art by enhancing existing over-activation techniques for real-world adversarial attacks to make them usable in real-time applications. It also introduces an efficient multi-frame defense framework, validating its efficacy through extensive experiments aimed at evaluating both defense performance and computational cost.



## **36. Boost Adversarial Transferability by Uniform Scale and Mix Mask Method**

cs.CV

**SubmitDate**: 2023-11-18    [abs](http://arxiv.org/abs/2311.12051v1) [paper-pdf](http://arxiv.org/pdf/2311.12051v1)

**Authors**: Tao Wang, Zijian Ying, Qianmu Li, zhichao Lian

**Abstract**: Adversarial examples generated from surrogate models often possess the ability to deceive other black-box models, a property known as transferability. Recent research has focused on enhancing adversarial transferability, with input transformation being one of the most effective approaches. However, existing input transformation methods suffer from two issues. Firstly, certain methods, such as the Scale-Invariant Method, employ exponentially decreasing scale invariant parameters that decrease the adaptability in generating effective adversarial examples across multiple scales. Secondly, most mixup methods only linearly combine candidate images with the source image, leading to reduced features blending effectiveness. To address these challenges, we propose a framework called Uniform Scale and Mix Mask Method (US-MM) for adversarial example generation. The Uniform Scale approach explores the upper and lower boundaries of perturbation with a linear factor, minimizing the negative impact of scale copies. The Mix Mask method introduces masks into the mixing process in a nonlinear manner, significantly improving the effectiveness of mixing strategies. Ablation experiments are conducted to validate the effectiveness of each component in US-MM and explore the effect of hyper-parameters. Empirical evaluations on standard ImageNet datasets demonstrate that US-MM achieves an average of 7% better transfer attack success rate compared to state-of-the-art methods.



## **37. Improving Adversarial Transferability by Stable Diffusion**

cs.CV

**SubmitDate**: 2023-11-18    [abs](http://arxiv.org/abs/2311.11017v1) [paper-pdf](http://arxiv.org/pdf/2311.11017v1)

**Authors**: Jiayang Liu, Siyu Zhu, Siyuan Liang, Jie Zhang, Han Fang, Weiming Zhang, Ee-Chien Chang

**Abstract**: Deep neural networks (DNNs) are susceptible to adversarial examples, which introduce imperceptible perturbations to benign samples, deceiving DNN predictions. While some attack methods excel in the white-box setting, they often struggle in the black-box scenario, particularly against models fortified with defense mechanisms. Various techniques have emerged to enhance the transferability of adversarial attacks for the black-box scenario. Among these, input transformation-based attacks have demonstrated their effectiveness. In this paper, we explore the potential of leveraging data generated by Stable Diffusion to boost adversarial transferability. This approach draws inspiration from recent research that harnessed synthetic data generated by Stable Diffusion to enhance model generalization. In particular, previous work has highlighted the correlation between the presence of both real and synthetic data and improved model generalization. Building upon this insight, we introduce a novel attack method called Stable Diffusion Attack Method (SDAM), which incorporates samples generated by Stable Diffusion to augment input images. Furthermore, we propose a fast variant of SDAM to reduce computational overhead while preserving high adversarial transferability. Our extensive experimental results demonstrate that our method outperforms state-of-the-art baselines by a substantial margin. Moreover, our approach is compatible with existing transfer-based attacks to further enhance adversarial transferability.



## **38. Security of quantum key distribution from generalised entropy accumulation**

quant-ph

30 pages

**SubmitDate**: 2023-11-18    [abs](http://arxiv.org/abs/2203.04993v2) [paper-pdf](http://arxiv.org/pdf/2203.04993v2)

**Authors**: Tony Metger, Renato Renner

**Abstract**: The goal of quantum key distribution (QKD) is to establish a secure key between two parties connected by an insecure quantum channel. To use a QKD protocol in practice, one has to prove that a finite size key is secure against general attacks: no matter the adversary's attack, they cannot gain useful information about the key. A much simpler task is to prove security against collective attacks, where the adversary is assumed to behave identically and independently in each round. In this work, we provide a formal framework for general QKD protocols and show that for any protocol that can be expressed in this framework, security against general attacks reduces to security against collective attacks, which in turn reduces to a numerical computation. Our proof relies on a recently developed information-theoretic tool called generalised entropy accumulation and can handle generic prepare-and-measure protocols directly without switching to an entanglement-based version.



## **39. PACOL: Poisoning Attacks Against Continual Learners**

cs.LG

**SubmitDate**: 2023-11-18    [abs](http://arxiv.org/abs/2311.10919v1) [paper-pdf](http://arxiv.org/pdf/2311.10919v1)

**Authors**: Huayu Li, Gregory Ditzler

**Abstract**: Continual learning algorithms are typically exposed to untrusted sources that contain training data inserted by adversaries and bad actors. An adversary can insert a small number of poisoned samples, such as mislabeled samples from previously learned tasks, or intentional adversarial perturbed samples, into the training datasets, which can drastically reduce the model's performance. In this work, we demonstrate that continual learning systems can be manipulated by malicious misinformation and present a new category of data poisoning attacks specific for continual learners, which we refer to as {\em Poisoning Attacks Against Continual Learners} (PACOL). The effectiveness of labeling flipping attacks inspires PACOL; however, PACOL produces attack samples that do not change the sample's label and produce an attack that causes catastrophic forgetting. A comprehensive set of experiments shows the vulnerability of commonly used generative replay and regularization-based continual learning approaches against attack methods. We evaluate the ability of label-flipping and a new adversarial poison attack, namely PACOL proposed in this work, to force the continual learning system to forget the knowledge of a learned task(s). More specifically, we compared the performance degradation of continual learning systems trained on benchmark data streams with and without poisoning attacks. Moreover, we discuss the stealthiness of the attacks in which we test the success rate of data sanitization defense and other outlier detection-based defenses for filtering out adversarial samples.



## **40. Parrot-Trained Adversarial Examples: Pushing the Practicality of Black-Box Audio Attacks against Speaker Recognition Models**

cs.SD

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2311.07780v2) [paper-pdf](http://arxiv.org/pdf/2311.07780v2)

**Authors**: Rui Duan, Zhe Qu, Leah Ding, Yao Liu, Zhuo Lu

**Abstract**: Audio adversarial examples (AEs) have posed significant security challenges to real-world speaker recognition systems. Most black-box attacks still require certain information from the speaker recognition model to be effective (e.g., keeping probing and requiring the knowledge of similarity scores). This work aims to push the practicality of the black-box attacks by minimizing the attacker's knowledge about a target speaker recognition model. Although it is not feasible for an attacker to succeed with completely zero knowledge, we assume that the attacker only knows a short (or a few seconds) speech sample of a target speaker. Without any probing to gain further knowledge about the target model, we propose a new mechanism, called parrot training, to generate AEs against the target model. Motivated by recent advancements in voice conversion (VC), we propose to use the one short sentence knowledge to generate more synthetic speech samples that sound like the target speaker, called parrot speech. Then, we use these parrot speech samples to train a parrot-trained(PT) surrogate model for the attacker. Under a joint transferability and perception framework, we investigate different ways to generate AEs on the PT model (called PT-AEs) to ensure the PT-AEs can be generated with high transferability to a black-box target model with good human perceptual quality. Real-world experiments show that the resultant PT-AEs achieve the attack success rates of 45.8% - 80.8% against the open-source models in the digital-line scenario and 47.9% - 58.3% against smart devices, including Apple HomePod (Siri), Amazon Echo, and Google Home, in the over-the-air scenario.



## **41. Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm**

cs.CV

8 pages, 3 figures

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2310.13019v3) [paper-pdf](http://arxiv.org/pdf/2310.13019v3)

**Authors**: S. M. Fazle Rabby Labib, Joyanta Jyoti Mondal, Meem Arafat Manab

**Abstract**: Deep neural networks (DNNs) have significantly advanced various domains, but their vulnerability to adversarial attacks poses serious concerns. Understanding these vulnerabilities and developing effective defense mechanisms is crucial. DeepFool, an algorithm proposed by Moosavi-Dezfooli et al. (2016), finds minimal perturbations to misclassify input images. However, DeepFool lacks a targeted approach, making it less effective in specific attack scenarios. Also, in previous related works, researchers primarily focus on success, not considering how much an image is getting distorted; the integrity of the image quality, and the confidence level to misclassifying. So, in this paper, we propose Enhanced Targeted DeepFool, an augmented version of DeepFool that allows targeting specific classes for misclassification and also introduce a minimum confidence score requirement hyperparameter to enhance flexibility. Our experiments demonstrate the effectiveness and efficiency of the proposed method across different deep neural network architectures while preserving image integrity as much and perturbation rate as less as possible. By using our approach, the behavior of models can be manipulated arbitrarily using the perturbed images, as we can specify both the target class and the associated confidence score, unlike other DeepFool-derivative works, such as Targeted DeepFool by Gajjar et al. (2022). Results show that one of the deep convolutional neural network architectures, AlexNet, and one of the state-of-the-art model Vision Transformer exhibit high robustness to getting fooled. This approach can have larger implication, as our tuning of confidence level can expose the robustness of image recognition models. Our code will be made public upon acceptance of the paper.



## **42. Breaking Boundaries: Balancing Performance and Robustness in Deep Wireless Traffic Forecasting**

cs.LG

Accepted for presentation at the ARTMAN workshop, part of the ACM  Conference on Computer and Communications Security (CCS), 2023

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2311.09790v2) [paper-pdf](http://arxiv.org/pdf/2311.09790v2)

**Authors**: Romain Ilbert, Thai V. Hoang, Zonghua Zhang, Themis Palpanas

**Abstract**: Balancing the trade-off between accuracy and robustness is a long-standing challenge in time series forecasting. While most of existing robust algorithms have achieved certain suboptimal performance on clean data, sustaining the same performance level in the presence of data perturbations remains extremely hard. In this paper, we study a wide array of perturbation scenarios and propose novel defense mechanisms against adversarial attacks using real-world telecom data. We compare our strategy against two existing adversarial training algorithms under a range of maximal allowed perturbations, defined using $\ell_{\infty}$-norm, $\in [0.1,0.4]$. Our findings reveal that our hybrid strategy, which is composed of a classifier to detect adversarial examples, a denoiser to eliminate noise from the perturbed data samples, and a standard forecaster, achieves the best performance on both clean and perturbed data. Our optimal model can retain up to $92.02\%$ the performance of the original forecasting model in terms of Mean Squared Error (MSE) on clean data, while being more robust than the standard adversarially trained models on perturbed data. Its MSE is 2.71$\times$ and 2.51$\times$ lower than those of comparing methods on normal and perturbed data, respectively. In addition, the components of our models can be trained in parallel, resulting in better computational efficiency. Our results indicate that we can optimally balance the trade-off between the performance and robustness of forecasting models by improving the classifier and denoiser, even in the presence of sophisticated and destructive poisoning attacks.



## **43. Laccolith: Hypervisor-Based Adversary Emulation with Anti-Detection**

cs.CR

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2311.08274v2) [paper-pdf](http://arxiv.org/pdf/2311.08274v2)

**Authors**: Vittorio Orbinato, Marco Carlo Feliciano, Domenico Cotroneo, Roberto Natella

**Abstract**: Advanced Persistent Threats (APTs) represent the most threatening form of attack nowadays since they can stay undetected for a long time. Adversary emulation is a proactive approach for preparing against these attacks. However, adversary emulation tools lack the anti-detection abilities of APTs. We introduce Laccolith, a hypervisor-based solution for adversary emulation with anti-detection to fill this gap. We also present an experimental study to compare Laccolith with MITRE CALDERA, a state-of-the-art solution for adversary emulation, against five popular anti-virus products. We found that CALDERA cannot evade detection, limiting the realism of emulated attacks, even when combined with a state-of-the-art anti-detection framework. Our experiments show that Laccolith can hide its activities from all the tested anti-virus products, thus making it suitable for realistic emulations.



## **44. Breaking Temporal Consistency: Generating Video Universal Adversarial Perturbations Using Image Models**

cs.CV

ICCV 2023

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2311.10366v1) [paper-pdf](http://arxiv.org/pdf/2311.10366v1)

**Authors**: Hee-Seon Kim, Minji Son, Minbeom Kim, Myung-Joon Kwon, Changick Kim

**Abstract**: As video analysis using deep learning models becomes more widespread, the vulnerability of such models to adversarial attacks is becoming a pressing concern. In particular, Universal Adversarial Perturbation (UAP) poses a significant threat, as a single perturbation can mislead deep learning models on entire datasets. We propose a novel video UAP using image data and image model. This enables us to take advantage of the rich image data and image model-based studies available for video applications. However, there is a challenge that image models are limited in their ability to analyze the temporal aspects of videos, which is crucial for a successful video attack. To address this challenge, we introduce the Breaking Temporal Consistency (BTC) method, which is the first attempt to incorporate temporal information into video attacks using image models. We aim to generate adversarial videos that have opposite patterns to the original. Specifically, BTC-UAP minimizes the feature similarity between neighboring frames in videos. Our approach is simple but effective at attacking unseen video models. Additionally, it is applicable to videos of varying lengths and invariant to temporal shifts. Our approach surpasses existing methods in terms of effectiveness on various datasets, including ImageNet, UCF-101, and Kinetics-400.



## **45. Quantum Public-Key Encryption with Tamper-Resilient Public Keys from One-Way Functions**

quant-ph

48 pages

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2304.01800v3) [paper-pdf](http://arxiv.org/pdf/2304.01800v3)

**Authors**: Fuyuki Kitagawa, Tomoyuki Morimae, Ryo Nishimaki, Takashi Yamakawa

**Abstract**: We construct quantum public-key encryption from one-way functions. In our construction, public keys are quantum, but ciphertexts are classical. Quantum public-key encryption from one-way functions (or weaker primitives such as pseudorandom function-like states) are also proposed in some recent works [Morimae-Yamakawa, eprint:2022/1336; Coladangelo, eprint:2023/282; Barooti-Grilo-Malavolta-Sattath-Vu-Walter, eprint:2023/877]. However, they have a huge drawback: they are secure only when quantum public keys can be transmitted to the sender (who runs the encryption algorithm) without being tampered with by the adversary, which seems to require unsatisfactory physical setup assumptions such as secure quantum channels. Our construction is free from such a drawback: it guarantees the secrecy of the encrypted messages even if we assume only unauthenticated quantum channels. Thus, the encryption is done with adversarially tampered quantum public keys. Our construction is the first quantum public-key encryption that achieves the goal of classical public-key encryption, namely, to establish secure communication over insecure channels, based only on one-way functions. Moreover, we show a generic compiler to upgrade security against chosen plaintext attacks (CPA security) into security against chosen ciphertext attacks (CCA security) only using one-way functions. As a result, we obtain CCA secure quantum public-key encryption based only on one-way functions.



## **46. You Cannot Escape Me: Detecting Evasions of SIEM Rules in Enterprise Networks**

cs.CR

To be published in Proceedings of the 33rd USENIX Security Symposium  (USENIX Security 2024)

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.10197v1) [paper-pdf](http://arxiv.org/pdf/2311.10197v1)

**Authors**: Rafael Uetz, Marco Herzog, Louis Hackländer, Simon Schwarz, Martin Henze

**Abstract**: Cyberattacks have grown into a major risk for organizations, with common consequences being data theft, sabotage, and extortion. Since preventive measures do not suffice to repel attacks, timely detection of successful intruders is crucial to stop them from reaching their final goals. For this purpose, many organizations utilize Security Information and Event Management (SIEM) systems to centrally collect security-related events and scan them for attack indicators using expert-written detection rules. However, as we show by analyzing a set of widespread SIEM detection rules, adversaries can evade almost half of them easily, allowing them to perform common malicious actions within an enterprise network without being detected. To remedy these critical detection blind spots, we propose the idea of adaptive misuse detection, which utilizes machine learning to compare incoming events to SIEM rules on the one hand and known-benign events on the other hand to discover successful evasions. Based on this idea, we present AMIDES, an open-source proof-of-concept adaptive misuse detection system. Using four weeks of SIEM events from a large enterprise network and more than 500 hand-crafted evasions, we show that AMIDES successfully detects a majority of these evasions without any false alerts. In addition, AMIDES eases alert analysis by assessing which rules were evaded. Its computational efficiency qualifies AMIDES for real-world operation and hence enables organizations to significantly reduce detection blind spots with moderate effort.



## **47. Differentiable JPEG: The Devil is in the Details**

cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2309.06978v2) [paper-pdf](http://arxiv.org/pdf/2309.06978v2)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.



## **48. Towards more Practical Threat Models in Artificial Intelligence Security**

cs.CR

18 pages, 4 figures, 7 tables, under submission

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09994v1) [paper-pdf](http://arxiv.org/pdf/2311.09994v1)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Alexandre Alahi

**Abstract**: Recent works have identified a gap between research and practice in artificial intelligence security: threats studied in academia do not always reflect the practical use and security risks of AI. For example, while models are often studied in isolation, they form part of larger ML pipelines in practice. Recent works also brought forward that adversarial manipulations introduced by academic attacks are impractical. We take a first step towards describing the full extent of this disparity. To this end, we revisit the threat models of the six most studied attacks in AI security research and match them to AI usage in practice via a survey with \textbf{271} industrial practitioners. On the one hand, we find that all existing threat models are indeed applicable. On the other hand, there are significant mismatches: research is often too generous with the attacker, assuming access to information not frequently available in real-world settings. Our paper is thus a call for action to study more practical threat models in artificial intelligence security.



## **49. Hijacking Large Language Models via Adversarial In-Context Learning**

cs.LG

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09948v1) [paper-pdf](http://arxiv.org/pdf/2311.09948v1)

**Authors**: Yao Qiang, Xiangyu Zhou, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific tasks by utilizing labeled examples as demonstrations in the precondition prompts. Despite its promising performance, ICL suffers from instability with the choice and arrangement of examples. Additionally, crafted adversarial attacks pose a notable threat to the robustness of ICL. However, existing attacks are either easy to detect, rely on external models, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable attack for ICL, aiming to hijack LLMs to generate the targeted response. The proposed LLM hijacking attack leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demonstrations. Extensive experimental results on various tasks and datasets demonstrate the effectiveness of our LLM hijacking attack, resulting in a distracted attention towards adversarial tokens, consequently leading to the targeted unwanted outputs.



## **50. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-start**

stat.ML

Corrected Remark 18 + other small edits. Code at  https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2202.03397v4) [paper-pdf](http://arxiv.org/pdf/2202.03397v4)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstract**: We analyse a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e.~they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise (near) optimal sample complexity. In particular, we propose a simple method which uses (stochastic) fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates.



