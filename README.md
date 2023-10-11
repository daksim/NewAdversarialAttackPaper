# Latest Adversarial Attack Papers
**update at 2023-10-11 17:11:35**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Graph-based methods coupled with specific distributional distances for adversarial attack detection**

cs.LG

published in Neural Networks

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2306.00042v2) [paper-pdf](http://arxiv.org/pdf/2306.00042v2)

**Authors**: Dwight Nwaigwe, Lucrezia Carboni, Martial Mermillod, Sophie Achard, Michel Dojat

**Abstract**: Artificial neural networks are prone to being fooled by carefully perturbed inputs which cause an egregious misclassification. These \textit{adversarial} attacks have been the focus of extensive research. Likewise, there has been an abundance of research in ways to detect and defend against them. We introduce a novel approach of detection and interpretation of adversarial attacks from a graph perspective. For an input image, we compute an associated sparse graph using the layer-wise relevance propagation algorithm \cite{bach15}. Specifically, we only keep edges of the neural network with the highest relevance values. Three quantities are then computed from the graph which are then compared against those computed from the training set. The result of the comparison is a classification of the image as benign or adversarial. To make the comparison, two classification methods are introduced: 1) an explicit formula based on Wasserstein distance applied to the degree of node and 2) a logistic regression. Both classification methods produce strong results which lead us to believe that a graph-based interpretation of adversarial attacks is valuable.



## **2. Privacy-oriented manipulation of speaker representations**

eess.AS

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06652v1) [paper-pdf](http://arxiv.org/pdf/2310.06652v1)

**Authors**: Francisco Teixeira, Alberto Abad, Bhiksha Raj, Isabel Trancoso

**Abstract**: Speaker embeddings are ubiquitous, with applications ranging from speaker recognition and diarization to speech synthesis and voice anonymisation. The amount of information held by these embeddings lends them versatility, but also raises privacy concerns. Speaker embeddings have been shown to contain information on age, sex, health and more, which speakers may want to keep private, especially when this information is not required for the target task. In this work, we propose a method for removing and manipulating private attributes from speaker embeddings that leverages a Vector-Quantized Variational Autoencoder architecture, combined with an adversarial classifier and a novel mutual information loss. We validate our model on two attributes, sex and age, and perform experiments with ignorant and fully-informed attackers, and with in-domain and out-of-domain data.



## **3. A Geometrical Approach to Evaluate the Adversarial Robustness of Deep Neural Networks**

cs.CV

ACM Transactions on Multimedia Computing, Communications, and  Applications (ACM TOMM)

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06468v1) [paper-pdf](http://arxiv.org/pdf/2310.06468v1)

**Authors**: Yang Wang, Bo Dong, Ke Xu, Haiyin Piao, Yufei Ding, Baocai Yin, Xin Yang

**Abstract**: Deep Neural Networks (DNNs) are widely used for computer vision tasks. However, it has been shown that deep models are vulnerable to adversarial attacks, i.e., their performances drop when imperceptible perturbations are made to the original inputs, which may further degrade the following visual tasks or introduce new problems such as data and privacy security. Hence, metrics for evaluating the robustness of deep models against adversarial attacks are desired. However, previous metrics are mainly proposed for evaluating the adversarial robustness of shallow networks on the small-scale datasets. Although the Cross Lipschitz Extreme Value for nEtwork Robustness (CLEVER) metric has been proposed for large-scale datasets (e.g., the ImageNet dataset), it is computationally expensive and its performance relies on a tractable number of samples. In this paper, we propose the Adversarial Converging Time Score (ACTS), an attack-dependent metric that quantifies the adversarial robustness of a DNN on a specific input. Our key observation is that local neighborhoods on a DNN's output surface would have different shapes given different inputs. Hence, given different inputs, it requires different time for converging to an adversarial sample. Based on this geometry meaning, ACTS measures the converging time as an adversarial robustness metric. We validate the effectiveness and generalization of the proposed ACTS metric against different adversarial attacks on the large-scale ImageNet dataset using state-of-the-art deep networks. Extensive experiments show that our ACTS metric is an efficient and effective adversarial metric over the previous CLEVER metric.



## **4. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.00322v2) [paper-pdf](http://arxiv.org/pdf/2310.00322v2)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.



## **5. Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach**

cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS), New Orleans, USA, Dec. 2023, spotlight

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06396v1) [paper-pdf](http://arxiv.org/pdf/2310.06396v1)

**Authors**: Kai Zhao, Qiyu Kang, Yang Song, Rui She, Sijie Wang, Wee Peng Tay

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial perturbations, including those that affect both node features and graph topology. This paper investigates GNNs derived from diverse neural flows, concentrating on their connection to various stability notions such as BIBO stability, Lyapunov stability, structural stability, and conservative stability. We argue that Lyapunov stability, despite its common use, does not necessarily ensure adversarial robustness. Inspired by physics principles, we advocate for the use of conservative Hamiltonian neural flows to construct GNNs that are robust to adversarial attacks. The adversarial robustness of different neural flow GNNs is empirically compared on several benchmark datasets under a variety of adversarial attacks. Extensive numerical experiments demonstrate that GNNs leveraging conservative Hamiltonian flows with Lyapunov stability substantially improve robustness against adversarial perturbations. The implementation code of experiments is available at https://github.com/zknus/NeurIPS-2023-HANG-Robustness.



## **6. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

cs.LG

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06387v1) [paper-pdf](http://arxiv.org/pdf/2310.06387v1)

**Authors**: Zeming Wei, Yifei Wang, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating malicious content have emerged. In this paper, we explore the power of In-Context Learning (ICL) in manipulating the alignment ability of LLMs. We find that by providing just few in-context demonstrations without fine-tuning, LLMs can be manipulated to increase or decrease the probability of jailbreaking, i.e. answering malicious prompts. Based on these observations, we propose In-Context Attack (ICA) and In-Context Defense (ICD) methods for jailbreaking and guarding aligned language model purposes. ICA crafts malicious contexts to guide models in generating harmful outputs, while ICD enhances model robustness by demonstrations of rejecting to answer harmful prompts. Our experiments show the effectiveness of ICA and ICD in increasing or reducing the success rate of adversarial jailbreaking attacks. Overall, we shed light on the potential of ICL to influence LLM behavior and provide a new perspective for enhancing the safety and alignment of LLMs.



## **7. Double Public Key Signing Function Oracle Attack on EdDSA Software Implementations**

cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2308.15009v2) [paper-pdf](http://arxiv.org/pdf/2308.15009v2)

**Authors**: Sam Grierson, Konstantinos Chalkias, William J Buchanan, Leandros Maglaras

**Abstract**: EdDSA is a standardised elliptic curve digital signature scheme introduced to overcome some of the issues prevalent in the more established ECDSA standard. Due to the EdDSA standard specifying that the EdDSA signature be deterministic, if the signing function were to be used as a public key signing oracle for the attacker, the unforgeability notion of security of the scheme can be broken. This paper describes an attack against some of the most popular EdDSA implementations, which results in an adversary recovering the private key used during signing. With this recovered secret key, an adversary can sign arbitrary messages that would be seen as valid by the EdDSA verification function. A list of libraries with vulnerable APIs at the time of publication is provided. Furthermore, this paper provides two suggestions for securing EdDSA signing APIs against this vulnerability while it additionally discusses failed attempts to solve the issue.



## **8. Exploring adversarial attacks in federated learning for medical imaging**

cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06227v1) [paper-pdf](http://arxiv.org/pdf/2310.06227v1)

**Authors**: Erfan Darzi, Florian Dubost, N. M. Sijtsema, P. M. A van Ooijen

**Abstract**: Federated learning offers a privacy-preserving framework for medical image analysis but exposes the system to adversarial attacks. This paper aims to evaluate the vulnerabilities of federated learning networks in medical image analysis against such attacks. Employing domain-specific MRI tumor and pathology imaging datasets, we assess the effectiveness of known threat scenarios in a federated learning environment. Our tests reveal that domain-specific configurations can increase the attacker's success rate significantly. The findings emphasize the urgent need for effective defense mechanisms and suggest a critical re-evaluation of current security protocols in federated medical image analysis systems.



## **9. PAC-Bayesian Spectrally-Normalized Bounds for Adversarially Robust Generalization**

cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.06182v1) [paper-pdf](http://arxiv.org/pdf/2310.06182v1)

**Authors**: Jiancong Xiao, Ruoyu Sun, Zhi-quan Luo

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. It is found empirically that adversarially robust generalization is crucial in establishing defense algorithms against adversarial attacks. Therefore, it is interesting to study the theoretical guarantee of robust generalization. This paper focuses on norm-based complexity, based on a PAC-Bayes approach (Neyshabur et al., 2017). The main challenge lies in extending the key ingredient, which is a weight perturbation bound in standard settings, to the robust settings. Existing attempts heavily rely on additional strong assumptions, leading to loose bounds. In this paper, we address this issue and provide a spectrally-normalized robust generalization bound for DNNs. Compared to existing bounds, our bound offers two significant advantages: Firstly, it does not depend on additional assumptions. Secondly, it is considerably tighter, aligning with the bounds of standard generalization. Therefore, our result provides a different perspective on understanding robust generalization: The mismatch terms between standard and robust generalization bounds shown in previous studies do not contribute to the poor robust generalization. Instead, these disparities solely due to mathematical issues. Finally, we extend the main result to adversarial robustness against general non-$\ell_p$ attacks and other neural network architectures.



## **10. Lessons Learned: Defending Against Property Inference Attacks**

cs.CR

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2205.08821v4) [paper-pdf](http://arxiv.org/pdf/2205.08821v4)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstract**: This work investigates and evaluates multiple defense strategies against property inference attacks (PIAs), a privacy attack against machine learning models. Given a trained machine learning model, PIAs aim to extract statistical properties of its underlying training data, e.g., reveal the ratio of men and women in a medical training data set. While for other privacy attacks like membership inference, a lot of research on defense mechanisms has been published, this is the first work focusing on defending against PIAs. With the primary goal of developing a generic mitigation strategy against white-box PIAs, we propose the novel approach property unlearning. Extensive experiments with property unlearning show that while it is very effective when defending target models against specific adversaries, property unlearning is not able to generalize, i.e., protect against a whole class of PIAs. To investigate the reasons behind this limitation, we present the results of experiments with the explainable AI tool LIME. They show how state-of-the-art property inference adversaries with the same objective focus on different parts of the target model. We further elaborate on this with a follow-up experiment, in which we use the visualization technique t-SNE to exhibit how severely statistical training data properties are manifested in machine learning models. Based on this, we develop the conjecture that post-training techniques like property unlearning might not suffice to provide the desirable generic protection against PIAs. As an alternative, we investigate the effects of simpler training data preprocessing methods like adding Gaussian noise to images of a training data set on the success rate of PIAs. We conclude with a discussion of the different defense approaches, summarize the lessons learned and provide directions for future work.



## **11. Universal adversarial perturbations for multiple classification tasks with quantum classifiers**

quant-ph

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2306.11974v2) [paper-pdf](http://arxiv.org/pdf/2306.11974v2)

**Authors**: Yun-Zhong Qiu

**Abstract**: Quantum adversarial machine learning is an emerging field that studies the vulnerability of quantum learning systems against adversarial perturbations and develops possible defense strategies. Quantum universal adversarial perturbations are small perturbations, which can make different input samples into adversarial examples that may deceive a given quantum classifier. This is a field that was rarely looked into but worthwhile investigating because universal perturbations might simplify malicious attacks to a large extent, causing unexpected devastation to quantum machine learning models. In this paper, we take a step forward and explore the quantum universal perturbations in the context of heterogeneous classification tasks. In particular, we find that quantum classifiers that achieve almost state-of-the-art accuracy on two different classification tasks can be both conclusively deceived by one carefully-crafted universal perturbation. This result is explicitly demonstrated with well-designed quantum continual learning models with elastic weight consolidation method to avoid catastrophic forgetting, as well as real-life heterogeneous datasets from hand-written digits and medical MRI images. Our results provide a simple and efficient way to generate universal perturbations on heterogeneous classification tasks and thus would provide valuable guidance for future quantum learning technologies.



## **12. RECESS Vaccine for Federated Learning: Proactive Defense Against Model Poisoning Attacks**

cs.CR

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05431v1) [paper-pdf](http://arxiv.org/pdf/2310.05431v1)

**Authors**: Haonan Yan, Wenjing Zhang, Qian Chen, Xiaoguang Li, Wenhai Sun, Hui Li, Xiaodong Lin

**Abstract**: Model poisoning attacks greatly jeopardize the application of federated learning (FL). The effectiveness of existing defenses is susceptible to the latest model poisoning attacks, leading to a decrease in prediction accuracy. Besides, these defenses are intractable to distinguish benign outliers from malicious gradients, which further compromises the model generalization. In this work, we propose a novel proactive defense named RECESS against model poisoning attacks. Different from the passive analysis in previous defenses, RECESS proactively queries each participating client with a delicately constructed aggregation gradient, accompanied by the detection of malicious clients according to their responses with higher accuracy. Furthermore, RECESS uses a new trust scoring mechanism to robustly aggregate gradients. Unlike previous methods that score each iteration, RECESS considers clients' performance correlation across multiple iterations to estimate the trust score, substantially increasing fault tolerance. Finally, we extensively evaluate RECESS on typical model architectures and four datasets under various settings. We also evaluated the defensive effectiveness against other types of poisoning attacks, the sensitivity of hyperparameters, and adaptive adversarial attacks. Experimental results show the superiority of RECESS in terms of reducing accuracy loss caused by the latest model poisoning attacks over five classic and two state-of-the-art defenses.



## **13. AdvSV: An Over-the-Air Adversarial Attack Dataset for Speaker Verification**

cs.SD

Submitted to ICASSP2024

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05369v1) [paper-pdf](http://arxiv.org/pdf/2310.05369v1)

**Authors**: Li Wang, Jiaqi Li, Yuhao Luo, Jiahao Zheng, Lei Wang, Hao Li, Ke Xu, Chengfang Fang, Jie Shi, Zhizheng Wu

**Abstract**: It is known that deep neural networks are vulnerable to adversarial attacks. Although Automatic Speaker Verification (ASV) built on top of deep neural networks exhibits robust performance in controlled scenarios, many studies confirm that ASV is vulnerable to adversarial attacks. The lack of a standard dataset is a bottleneck for further research, especially reproducible research. In this study, we developed an open-source adversarial attack dataset for speaker verification research. As an initial step, we focused on the over-the-air attack. An over-the-air adversarial attack involves a perturbation generation algorithm, a loudspeaker, a microphone, and an acoustic environment. The variations in the recording configurations make it very challenging to reproduce previous research. The AdvSV dataset is constructed using the Voxceleb1 Verification test set as its foundation. This dataset employs representative ASV models subjected to adversarial attacks and records adversarial samples to simulate over-the-air attack settings. The scope of the dataset can be easily extended to include more types of adversarial attacks. The dataset will be released to the public under the CC-BY license. In addition, we also provide a detection baseline for reproducible research.



## **14. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

cs.SD

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05354v1) [paper-pdf](http://arxiv.org/pdf/2310.05354v1)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.



## **15. GReAT: A Graph Regularized Adversarial Training Method**

cs.LG

25 pages including references. 7 figures and 4 tables

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05336v1) [paper-pdf](http://arxiv.org/pdf/2310.05336v1)

**Authors**: Samet Bayram, Kenneth Barner

**Abstract**: This paper proposes a regularization method called GReAT, Graph Regularized Adversarial Training, to improve deep learning models' classification performance. Adversarial examples are a well-known challenge in machine learning, where small, purposeful perturbations to input data can mislead models. Adversarial training, a powerful and one of the most effective defense strategies, involves training models with both regular and adversarial examples. However, it often neglects the underlying structure of the data. In response, we propose GReAT, a method that leverages data graph structure to enhance model robustness. GReAT deploys the graph structure of the data into the adversarial training process, resulting in more robust models that better generalize its testing performance and defend against adversarial attacks. Through extensive evaluation on benchmark datasets, we demonstrate GReAT's effectiveness compared to state-of-the-art classification methods, highlighting its potential in improving deep learning models' classification performance.



## **16. On the Query Complexity of Training Data Reconstruction in Private Learning**

cs.LG

Matching upper bounds, new corollaries for DP variants

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2303.16372v5) [paper-pdf](http://arxiv.org/pdf/2303.16372v5)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We analyze the number of queries that a whitebox adversary needs to make to a private learner in order to reconstruct its training data. For $(\epsilon, \delta)$ DP learners with training data drawn from any arbitrary compact metric space, we provide the \emph{first known lower bounds on the adversary's query complexity} as a function of the learner's privacy parameters. \emph{Our results are minimax optimal for every $\epsilon \geq 0, \delta \in [0, 1]$, covering both $\epsilon$-DP and $(0, \delta)$ DP as corollaries}. Beyond this, we obtain query complexity lower bounds for $(\alpha, \epsilon)$ R\'enyi DP learners that are valid for any $\alpha > 1, \epsilon \geq 0$. Finally, we analyze data reconstruction attacks on locally compact metric spaces via the framework of Metric DP, a generalization of DP that accounts for the underlying metric structure of the data. In this setting, we provide the first known analysis of data reconstruction in unbounded, high dimensional spaces and obtain query complexity lower bounds that are nearly tight modulo logarithmic factors.



## **17. Adversarial Attacks on Combinatorial Multi-Armed Bandits**

cs.LG

28 pages

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2310.05308v1) [paper-pdf](http://arxiv.org/pdf/2310.05308v1)

**Authors**: Rishab Balasubramanian, Jiawei Li, Prasad Tadepalli, Huazheng Wang, Qingyun Wu, Haoyu Zhao

**Abstract**: We study reward poisoning attacks on Combinatorial Multi-armed Bandits (CMAB). We first provide a sufficient and necessary condition for the attackability of CMAB, which depends on the intrinsic properties of the corresponding CMAB instance such as the reward distributions of super arms and outcome distributions of base arms. Additionally, we devise an attack algorithm for attackable CMAB instances. Contrary to prior understanding of multi-armed bandits, our work reveals a surprising fact that the attackability of a specific CMAB instance also depends on whether the bandit instance is known or unknown to the adversary. This finding indicates that adversarial attacks on CMAB are difficult in practice and a general attack strategy for any CMAB instance does not exist since the environment is mostly unknown to the adversary. We validate our theoretical findings via extensive experiments on real-world CMAB applications including probabilistic maximum covering problem, online minimum spanning tree, cascading bandits for online ranking, and online shortest path.



## **18. Robust Lipschitz Bandits to Adversarial Corruptions**

cs.LG

Thirty-seventh Conference on Neural Information Processing Systems  (NeurIPS 2023)

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2305.18543v2) [paper-pdf](http://arxiv.org/pdf/2305.18543v2)

**Authors**: Yue Kang, Cho-Jui Hsieh, Thomas C. M. Lee

**Abstract**: Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.



## **19. Susceptibility of Continual Learning Against Adversarial Attacks**

cs.LG

18 pages, 13 figures

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2207.05225v5) [paper-pdf](http://arxiv.org/pdf/2207.05225v5)

**Authors**: Hikmat Khan, Pir Masoom Shah, Syed Farhan Alam Zaidi, Saif ul Islam, Qasim Zia

**Abstract**: Recent continual learning approaches have primarily focused on mitigating catastrophic forgetting. Nevertheless, two critical areas have remained relatively unexplored: 1) evaluating the robustness of proposed methods and 2) ensuring the security of learned tasks. This paper investigates the susceptibility of continually learned tasks, including current and previously acquired tasks, to adversarial attacks. Specifically, we have observed that any class belonging to any task can be easily targeted and misclassified as the desired target class of any other task. Such susceptibility or vulnerability of learned tasks to adversarial attacks raises profound concerns regarding data integrity and privacy. To assess the robustness of continual learning approaches, we consider continual learning approaches in all three scenarios, i.e., task-incremental learning, domain-incremental learning, and class-incremental learning. In this regard, we explore the robustness of three regularization-based methods, three replay-based approaches, and one hybrid technique that combines replay and exemplar approaches. We empirically demonstrated that in any setting of continual learning, any class, whether belonging to the current or previously learned tasks, is susceptible to misclassification. Our observations identify potential limitations of continual learning approaches against adversarial attacks and highlight that current continual learning algorithms could not be suitable for deployment in real-world settings.



## **20. Transferable Availability Poisoning Attacks**

cs.CR

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2310.05141v1) [paper-pdf](http://arxiv.org/pdf/2310.05141v1)

**Authors**: Yiyong Liu, Michael Backes, Xiao Zhang

**Abstract**: We consider availability data poisoning attacks, where an adversary aims to degrade the overall test accuracy of a machine learning model by crafting small perturbations to its training data. Existing poisoning strategies can achieve the attack goal but assume the victim to employ the same learning method as what the adversary uses to mount the attack. In this paper, we argue that this assumption is strong, since the victim may choose any learning algorithm to train the model as long as it can achieve some targeted performance on clean data. Empirically, we observe a large decrease in the effectiveness of prior poisoning attacks if the victim uses a different learning paradigm to train the model and show marked differences in frequency-level characteristics between perturbations generated with respect to different learners and attack methods. To enhance the attack transferability, we propose Transferable Poisoning, which generates high-frequency poisoning perturbations by alternately leveraging the gradient information with two specific algorithms selected from supervised and unsupervised contrastive learning paradigms. Through extensive experiments on benchmark image datasets, we show that our transferable poisoning attack can produce poisoned samples with significantly improved transferability, not only applicable to the two learners used to devise the attack but also for learning algorithms and even paradigms beyond.



## **21. Model Extraction Attack against Self-supervised Speech Models**

cs.SD

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2211.16044v2) [paper-pdf](http://arxiv.org/pdf/2211.16044v2)

**Authors**: Tsu-Yuan Hsu, Chen-An Li, Tung-Yu Wu, Hung-yi Lee

**Abstract**: Self-supervised learning (SSL) speech models generate meaningful representations of given clips and achieve incredible performance across various downstream tasks. Model extraction attack (MEA) often refers to an adversary stealing the functionality of the victim model with only query access. In this work, we study the MEA problem against SSL speech model with a small number of queries. We propose a two-stage framework to extract the model. In the first stage, SSL is conducted on the large-scale unlabeled corpus to pre-train a small speech model. Secondly, we actively sample a small portion of clips from the unlabeled corpus and query the target model with these clips to acquire their representations as labels for the small model's second-stage training. Experiment results show that our sampling methods can effectively extract the target model without knowing any information about its model architecture.



## **22. An Anomaly Behavior Analysis Framework for Securing Autonomous Vehicle Perception**

cs.RO

20th ACS/IEEE International Conference on Computer Systems and  Applications (Accepted for publication)

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2310.05041v1) [paper-pdf](http://arxiv.org/pdf/2310.05041v1)

**Authors**: Murad Mehrab Abrar, Salim Hariri

**Abstract**: As a rapidly growing cyber-physical platform, Autonomous Vehicles (AVs) are encountering more security challenges as their capabilities continue to expand. In recent years, adversaries are actively targeting the perception sensors of autonomous vehicles with sophisticated attacks that are not easily detected by the vehicles' control systems. This work proposes an Anomaly Behavior Analysis approach to detect a perception sensor attack against an autonomous vehicle. The framework relies on temporal features extracted from a physics-based autonomous vehicle behavior model to capture the normal behavior of vehicular perception in autonomous driving. By employing a combination of model-based techniques and machine learning algorithms, the proposed framework distinguishes between normal and abnormal vehicular perception behavior. To demonstrate the application of the framework in practice, we performed a depth camera attack experiment on an autonomous vehicle testbed and generated an extensive dataset. We validated the effectiveness of the proposed framework using this real-world data and released the dataset for public access. To our knowledge, this dataset is the first of its kind and will serve as a valuable resource for the research community in evaluating their intrusion detection techniques effectively.



## **23. Robust Network Pruning With Sparse Entropic Wasserstein Regression**

cs.AI

submitted to ICLR 2024

**SubmitDate**: 2023-10-07    [abs](http://arxiv.org/abs/2310.04918v1) [paper-pdf](http://arxiv.org/pdf/2310.04918v1)

**Authors**: Lei You, Hei Victor Cheng

**Abstract**: This study unveils a cutting-edge technique for neural network pruning that judiciously addresses noisy gradients during the computation of the empirical Fisher Information Matrix (FIM). We introduce an entropic Wasserstein regression (EWR) formulation, capitalizing on the geometric attributes of the optimal transport (OT) problem. This is analytically showcased to excel in noise mitigation by adopting neighborhood interpolation across data points. The unique strength of the Wasserstein distance is its intrinsic ability to strike a balance between noise reduction and covariance information preservation. Extensive experiments performed on various networks show comparable performance of the proposed method with state-of-the-art (SoTA) network pruning algorithms. Our proposed method outperforms the SoTA when the network size or the target sparsity is large, the gain is even larger with the existence of noisy gradients, possibly from noisy data, analog memory, or adversarial attacks. Notably, our proposed method achieves a gain of 6% improvement in accuracy and 8% improvement in testing loss for MobileNetV1 with less than one-fourth of the network parameters remaining.



## **24. A Survey of Graph Unlearning**

cs.LG

22 page review paper on graph unlearning

**SubmitDate**: 2023-10-07    [abs](http://arxiv.org/abs/2310.02164v2) [paper-pdf](http://arxiv.org/pdf/2310.02164v2)

**Authors**: Anwar Said, Tyler Derr, Mudassir Shabbir, Waseem Abbas, Xenofon Koutsoukos

**Abstract**: Graph unlearning emerges as a crucial advancement in the pursuit of responsible AI, providing the means to remove sensitive data traces from trained models, thereby upholding the right to be forgotten. It is evident that graph machine learning exhibits sensitivity to data privacy and adversarial attacks, necessitating the application of graph unlearning techniques to address these concerns effectively. In this comprehensive survey paper, we present the first systematic review of graph unlearning approaches, encompassing a diverse array of methodologies and offering a detailed taxonomy and up-to-date literature overview to facilitate the understanding of researchers new to this field. Additionally, we establish the vital connections between graph unlearning and differential privacy, augmenting our understanding of the relevance of privacy-preserving techniques in this context. To ensure clarity, we provide lucid explanations of the fundamental concepts and evaluation measures used in graph unlearning, catering to a broader audience with varying levels of expertise. Delving into potential applications, we explore the versatility of graph unlearning across various domains, including but not limited to social networks, adversarial settings, and resource-constrained environments like the Internet of Things (IoT), illustrating its potential impact in safeguarding data privacy and enhancing AI systems' robustness. Finally, we shed light on promising research directions, encouraging further progress and innovation within the domain of graph unlearning. By laying a solid foundation and fostering continued progress, this survey seeks to inspire researchers to further advance the field of graph unlearning, thereby instilling confidence in the ethical growth of AI systems and reinforcing the responsible application of machine learning techniques in various domains.



## **25. Untargeted White-box Adversarial Attack with Heuristic Defence Methods in Real-time Deep Learning based Network Intrusion Detection System**

cs.LG

**SubmitDate**: 2023-10-07    [abs](http://arxiv.org/abs/2310.03334v2) [paper-pdf](http://arxiv.org/pdf/2310.03334v2)

**Authors**: Khushnaseeb Roshan, Aasim Zafar, Sheikh Burhan Ul Haque

**Abstract**: Network Intrusion Detection System (NIDS) is a key component in securing the computer network from various cyber security threats and network attacks. However, consider an unfortunate situation where the NIDS is itself attacked and vulnerable more specifically, we can say, How to defend the defender?. In Adversarial Machine Learning (AML), the malicious actors aim to fool the Machine Learning (ML) and Deep Learning (DL) models to produce incorrect predictions with intentionally crafted adversarial examples. These adversarial perturbed examples have become the biggest vulnerability of ML and DL based systems and are major obstacles to their adoption in real-time and mission-critical applications such as NIDS. AML is an emerging research domain, and it has become a necessity for the in-depth study of adversarial attacks and their defence strategies to safeguard the computer network from various cyber security threads. In this research work, we aim to cover important aspects related to NIDS, adversarial attacks and its defence mechanism to increase the robustness of the ML and DL based NIDS. We implemented four powerful adversarial attack techniques, namely, Fast Gradient Sign Method (FGSM), Jacobian Saliency Map Attack (JSMA), Projected Gradient Descent (PGD) and Carlini & Wagner (C&W) in NIDS. We analyzed its performance in terms of various performance metrics in detail. Furthermore, the three heuristics defence strategies, i.e., Adversarial Training (AT), Gaussian Data Augmentation (GDA) and High Confidence (HC), are implemented to improve the NIDS robustness under adversarial attack situations. The complete workflow is demonstrated in real-time network with data packet flow. This research work provides the overall background for the researchers interested in AML and its implementation from a computer network security point of view.



## **26. Understanding and Improving Adversarial Attacks on Latent Diffusion Model**

cs.CV

**SubmitDate**: 2023-10-07    [abs](http://arxiv.org/abs/2310.04687v1) [paper-pdf](http://arxiv.org/pdf/2310.04687v1)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu

**Abstract**: Latent Diffusion Model (LDM) has emerged as a leading tool in image generation, particularly with its capability in few-shot generation. This capability also presents risks, notably in unauthorized artwork replication and misinformation generation. In response, adversarial attacks have been designed to safeguard personal images from being used as reference data. However, existing adversarial attacks are predominantly empirical, lacking a solid theoretical foundation. In this paper, we introduce a comprehensive theoretical framework for understanding adversarial attacks on LDM. Based on the framework, we propose a novel adversarial attack that exploits a unified target to guide the adversarial attack both in the forward and the reverse process of LDM. We provide empirical evidences that our method overcomes the offset problem of the optimization of adversarial attacks in existing methods. Through rigorous experiments, our findings demonstrate that our method outperforms current attacks and is able to generalize over different state-of-the-art few-shot generation pipelines based on LDM. Our method can serve as a stronger and efficient tool for people exposed to the risk of data privacy and security to protect themselves in the new era of powerful generative models. The code is available on GitHub: https://github.com/CaradryanLiang/ImprovedAdvDM.git.



## **27. VLAttack: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models**

cs.CR

Accepted by NeurIPS 2023

**SubmitDate**: 2023-10-07    [abs](http://arxiv.org/abs/2310.04655v1) [paper-pdf](http://arxiv.org/pdf/2310.04655v1)

**Authors**: Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma

**Abstract**: Vision-Language (VL) pre-trained models have shown their superiority on many multimodal tasks. However, the adversarial robustness of such models has not been fully explored. Existing approaches mainly focus on exploring the adversarial robustness under the white-box setting, which is unrealistic. In this paper, we aim to investigate a new yet practical task to craft image and text perturbations using pre-trained VL models to attack black-box fine-tuned models on different downstream tasks. Towards this end, we propose VLAttack to generate adversarial samples by fusing perturbations of images and texts from both single-modal and multimodal levels. At the single-modal level, we propose a new block-wise similarity attack (BSA) strategy to learn image perturbations for disrupting universal representations. Besides, we adopt an existing text attack strategy to generate text perturbations independent of the image-modal attack. At the multimodal level, we design a novel iterative cross-search attack (ICSA) method to update adversarial image-text pairs periodically, starting with the outputs from the single-modal level. We conduct extensive experiments to attack three widely-used VL pretrained models for six tasks on eight datasets. Experimental results show that the proposed VLAttack framework achieves the highest attack success rates on all tasks compared with state-of-the-art baselines, which reveals a significant blind spot in the deployment of pre-trained VL models. Codes will be released soon.



## **28. RETVec: Resilient and Efficient Text Vectorizer**

cs.CL

Accepted at NeurIPS 2023

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2302.09207v2) [paper-pdf](http://arxiv.org/pdf/2302.09207v2)

**Authors**: Elie Bursztein, Marina Zhang, Owen Vallis, Xinyu Jia, Alexey Kurakin

**Abstract**: This paper describes RETVec, an efficient, resilient, and multilingual text vectorizer designed for neural-based text processing. RETVec combines a novel character encoding with an optional small embedding model to embed words into a 256-dimensional vector space. The RETVec embedding model is pre-trained using pair-wise metric learning to be robust against typos and character-level adversarial attacks. In this paper, we evaluate and compare RETVec to state-of-the-art vectorizers and word embeddings on popular model architectures and datasets. These comparisons demonstrate that RETVec leads to competitive, multilingual models that are significantly more resilient to typos and adversarial text attacks. RETVec is available under the Apache 2 license at https://github.com/google-research/retvec.



## **29. Adjustable Robust Reinforcement Learning for Online 3D Bin Packing**

cs.LG

Accepted to NeurIPS2023

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2310.04323v1) [paper-pdf](http://arxiv.org/pdf/2310.04323v1)

**Authors**: Yuxin Pan, Yize Chen, Fangzhen Lin

**Abstract**: Designing effective policies for the online 3D bin packing problem (3D-BPP) has been a long-standing challenge, primarily due to the unpredictable nature of incoming box sequences and stringent physical constraints. While current deep reinforcement learning (DRL) methods for online 3D-BPP have shown promising results in optimizing average performance over an underlying box sequence distribution, they often fail in real-world settings where some worst-case scenarios can materialize. Standard robust DRL algorithms tend to overly prioritize optimizing the worst-case performance at the expense of performance under normal problem instance distribution. To address these issues, we first introduce a permutation-based attacker to investigate the practical robustness of both DRL-based and heuristic methods proposed for solving online 3D-BPP. Then, we propose an adjustable robust reinforcement learning (AR2L) framework that allows efficient adjustment of robustness weights to achieve the desired balance of the policy's performance in average and worst-case environments. Specifically, we formulate the objective function as a weighted sum of expected and worst-case returns, and derive the lower performance bound by relating to the return under a mixture dynamics. To realize this lower bound, we adopt an iterative procedure that searches for the associated mixture dynamics and improves the corresponding policy. We integrate this procedure into two popular robust adversarial algorithms to develop the exact and approximate AR2L algorithms. Experiments demonstrate that AR2L is versatile in the sense that it improves policy robustness while maintaining an acceptable level of performance for the nominal case.



## **30. Assessing Robustness via Score-Based Adversarial Image Generation**

cs.CV

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2310.04285v1) [paper-pdf](http://arxiv.org/pdf/2310.04285v1)

**Authors**: Marcel Kollovieh, Lukas Gosch, Yan Scholten, Marten Lienen, Stephan Günnemann

**Abstract**: Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantic-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate adversarial examples beyond $\ell_p$-norm constraints, so-called unrestricted adversarial examples, overcoming their limitations. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating realistic adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG matches the performance of state-of-the-art attacks and defenses across multiple benchmarks. This work highlights the importance of investigating adversarial examples bounded by semantics rather than $\ell_p$-norm constraints. ScoreAG represents an important step towards more encompassing robustness assessments.



## **31. Threat Trekker: An Approach to Cyber Threat Hunting**

cs.CR

I am disseminating this outcome to all of you, despite the fact that  the results may appear somewhat idealistic, given that certain datasets  utilized for the training of the machine learning model comprise simulated  data

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2310.04197v1) [paper-pdf](http://arxiv.org/pdf/2310.04197v1)

**Authors**: Ángel Casanova Bienzobas, Alfonso Sánchez-Macián

**Abstract**: Threat hunting is a proactive methodology for exploring, detecting and mitigating cyberattacks within complex environments. As opposed to conventional detection systems, threat hunting strategies assume adversaries have infiltrated the system; as a result they proactively search out any unusual patterns or activities which might indicate intrusion attempts.   Historically, this endeavour has been pursued using three investigation methodologies: (1) Hypothesis-Driven Investigations; (2) Indicator of Compromise (IOC); and (3) High-level machine learning analysis-based approaches. Therefore, this paper introduces a novel machine learning paradigm known as Threat Trekker. This proposal utilizes connectors to feed data directly into an event streaming channel for processing by the algorithm and provide feedback back into its host network.   Conclusions drawn from these experiments clearly establish the efficacy of employing machine learning for classifying more subtle attacks.



## **32. Adversarial Illusions in Multi-Modal Embeddings**

cs.CR

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2308.11804v2) [paper-pdf](http://arxiv.org/pdf/2308.11804v2)

**Authors**: Eugene Bagdasaryan, Rishi Jha, Tingwei Zhang, Vitaly Shmatikov

**Abstract**: Multi-modal embeddings encode images, sounds, texts, videos, etc. into a single embedding space, aligning representations across modalities (e.g., associate an image of a dog with a barking sound). We show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an image or a sound, an adversary can perturb it so as to make its embedding close to an arbitrary, adversary-chosen input in another modality. This enables the adversary to align any image and any sound with any text.   Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks. Using ImageBind embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, and zero-shot classification.



## **33. FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and Federated LLMs**

cs.CR

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2306.04959v3) [paper-pdf](http://arxiv.org/pdf/2306.04959v3)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedMLSecurity, a benchmark designed to simulate adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). As an integral module of the open-sourced library FedML that facilitates FL algorithm development and performance comparison, FedMLSecurity enhances FedML's capabilities to evaluate security issues and potential remedies in FL. FedMLSecurity comprises two major components: FedMLAttacker that simulates attacks injected during FL training, and FedMLDefender that simulates defensive mechanisms to mitigate the impacts of the attacks. FedMLSecurity is open-sourced and can be customized to a wide range of machine learning models (e.g., Logistic Regression, ResNet, GAN, etc.) and federated optimizers (e.g., FedAVG, FedOPT, FedNOVA, etc.). FedMLSecurity can also be applied to Large Language Models (LLMs) easily, demonstrating its adaptability and applicability in various scenarios.



## **34. Kick Bad Guys Out! Zero-Knowledge-Proof-Based Anomaly Detection in Federated Learning**

cs.CR

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2310.04055v1) [paper-pdf](http://arxiv.org/pdf/2310.04055v1)

**Authors**: Shanshan Han, Wenxuan Wu, Baturalp Buyukates, Weizhao Jin, Yuhang Yao, Qifan Zhang, Salman Avestimehr, Chaoyang He

**Abstract**: Federated learning (FL) systems are vulnerable to malicious clients that submit poisoned local models to achieve their adversarial goals, such as preventing the convergence of the global model or inducing the global model to misclassify some data. Many existing defense mechanisms are impractical in real-world FL systems, as they require prior knowledge of the number of malicious clients or rely on re-weighting or modifying submissions. This is because adversaries typically do not announce their intentions before attacking, and re-weighting might change aggregation results even in the absence of attacks. To address these challenges in real FL systems, this paper introduces a cutting-edge anomaly detection approach with the following features: i) Detecting the occurrence of attacks and performing defense operations only when attacks happen; ii) Upon the occurrence of an attack, further detecting the malicious client models and eliminating them without harming the benign ones; iii) Ensuring honest execution of defense mechanisms at the server by leveraging a zero-knowledge proof mechanism. We validate the superior performance of the proposed approach with extensive experiments.



## **35. Improving classifier decision boundaries using nearest neighbors**

cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03927v1) [paper-pdf](http://arxiv.org/pdf/2310.03927v1)

**Authors**: Johannes Schneider

**Abstract**: Neural networks are not learning optimal decision boundaries. We show that decision boundaries are situated in areas of low training data density. They are impacted by few training samples which can easily lead to overfitting. We provide a simple algorithm performing a weighted average of the prediction of a sample and its nearest neighbors' (computed in latent space) leading to a minor favorable outcomes for a variety of important measures for neural networks. In our evaluation, we employ various self-trained and pre-trained convolutional neural networks to show that our approach improves (i) resistance to label noise, (ii) robustness against adversarial attacks, (iii) classification accuracy, and to some degree even (iv) interpretability. While improvements are not necessarily large in all four areas, our approach is conceptually simple, i.e., improvements come without any modification to network architecture, training procedure or dataset. Furthermore, they are in stark contrast to prior works that often require trade-offs among the four objectives or provide valuable, but non-actionable insights.



## **36. Preserving Semantics in Textual Adversarial Attacks**

cs.CL

8 pages, 4 figures

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2211.04205v2) [paper-pdf](http://arxiv.org/pdf/2211.04205v2)

**Authors**: David Herel, Hugo Cisneros, Tomas Mikolov

**Abstract**: The growth of hateful online content, or hate speech, has been associated with a global increase in violent crimes against minorities [23]. Harmful online content can be produced easily, automatically and anonymously. Even though, some form of auto-detection is already achieved through text classifiers in NLP, they can be fooled by adversarial attacks. To strengthen existing systems and stay ahead of attackers, we need better adversarial attacks. In this paper, we show that up to 70% of adversarial examples generated by adversarial attacks should be discarded because they do not preserve semantics. We address this core weakness and propose a new, fully supervised sentence embedding technique called Semantics-Preserving-Encoder (SPE). Our method outperforms existing sentence encoders used in adversarial attacks by achieving 1.2x - 5.1x better real attack success rate. We release our code as a plugin that can be used in any existing adversarial attack to improve its quality and speed up its execution.



## **37. OMG-ATTACK: Self-Supervised On-Manifold Generation of Transferable Evasion Attacks**

cs.LG

ICCV 2023, AROW Workshop

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03707v1) [paper-pdf](http://arxiv.org/pdf/2310.03707v1)

**Authors**: Ofir Bar Tal, Adi Haviv, Amit H. Bermano

**Abstract**: Evasion Attacks (EA) are used to test the robustness of trained neural networks by distorting input data to misguide the model into incorrect classifications. Creating these attacks is a challenging task, especially with the ever-increasing complexity of models and datasets. In this work, we introduce a self-supervised, computationally economical method for generating adversarial examples, designed for the unseen black-box setting. Adapting techniques from representation learning, our method generates on-manifold EAs that are encouraged to resemble the data distribution. These attacks are comparable in effectiveness compared to the state-of-the-art when attacking the model trained on, but are significantly more effective when attacking unseen models, as the attacks are more related to the data rather than the model itself. Our experiments consistently demonstrate the method is effective across various models, unseen data categories, and even defended models, suggesting a significant role for on-manifold EAs when targeting unseen models.



## **38. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03684v1) [paper-pdf](http://arxiv.org/pdf/2310.03684v1)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.



## **39. Certification of Deep Learning Models for Medical Image Segmentation**

eess.IV

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03664v1) [paper-pdf](http://arxiv.org/pdf/2310.03664v1)

**Authors**: Othmane Laousy, Alexandre Araujo, Guillaume Chassagnon, Nikos Paragios, Marie-Pierre Revel, Maria Vakalopoulou

**Abstract**: In medical imaging, segmentation models have known a significant improvement in the past decade and are now used daily in clinical practice. However, similar to classification models, segmentation models are affected by adversarial attacks. In a safety-critical field like healthcare, certifying model predictions is of the utmost importance. Randomized smoothing has been introduced lately and provides a framework to certify models and obtain theoretical guarantees. In this paper, we present for the first time a certified segmentation baseline for medical imaging based on randomized smoothing and diffusion models. Our results show that leveraging the power of denoising diffusion probabilistic models helps us overcome the limits of randomized smoothing. We conduct extensive experiments on five public datasets of chest X-rays, skin lesions, and colonoscopies, and empirically show that we are able to maintain high certified Dice scores even for highly perturbed images. Our work represents the first attempt to certify medical image segmentation models, and we aspire for it to set a foundation for future benchmarks in this crucial and largely uncharted area.



## **40. Targeted Adversarial Attacks on Generalizable Neural Radiance Fields**

cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03578v1) [paper-pdf](http://arxiv.org/pdf/2310.03578v1)

**Authors**: Andras Horvath, Csaba M. Jozsa

**Abstract**: Neural Radiance Fields (NeRFs) have recently emerged as a powerful tool for 3D scene representation and rendering. These data-driven models can learn to synthesize high-quality images from sparse 2D observations, enabling realistic and interactive scene reconstructions. However, the growing usage of NeRFs in critical applications such as augmented reality, robotics, and virtual environments could be threatened by adversarial attacks.   In this paper we present how generalizable NeRFs can be attacked by both low-intensity adversarial attacks and adversarial patches, where the later could be robust enough to be used in real world applications. We also demonstrate targeted attacks, where a specific, predefined output scene is generated by these attack with success.



## **41. Enhancing Adversarial Robustness via Score-Based Optimization**

cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2307.04333v2) [paper-pdf](http://arxiv.org/pdf/2307.04333v2)

**Authors**: Boya Zhang, Weijian Luo, Zhihua Zhang

**Abstract**: Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.



## **42. AdvRain: Adversarial Raindrops to Attack Camera-based Smart Vision Systems**

cs.CV

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2303.01338v2) [paper-pdf](http://arxiv.org/pdf/2303.01338v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Muhammad Shafique

**Abstract**: Vision-based perception modules are increasingly deployed in many applications, especially autonomous vehicles and intelligent robots. These modules are being used to acquire information about the surroundings and identify obstacles. Hence, accurate detection and classification are essential to reach appropriate decisions and take appropriate and safe actions at all times. Current studies have demonstrated that "printed adversarial attacks", known as physical adversarial attacks, can successfully mislead perception models such as object detectors and image classifiers. However, most of these physical attacks are based on noticeable and eye-catching patterns for generated perturbations making them identifiable/detectable by human eye or in test drives. In this paper, we propose a camera-based inconspicuous adversarial attack (\textbf{AdvRain}) capable of fooling camera-based perception systems over all objects of the same class. Unlike mask based fake-weather attacks that require access to the underlying computing hardware or image memory, our attack is based on emulating the effects of a natural weather condition (i.e., Raindrops) that can be printed on a translucent sticker, which is externally placed over the lens of a camera. To accomplish this, we provide an iterative process based on performing a random search aiming to identify critical positions to make sure that the performed transformation is adversarial for a target classifier. Our transformation is based on blurring predefined parts of the captured image corresponding to the areas covered by the raindrop. We achieve a drop in average model accuracy of more than $45\%$ and $40\%$ on VGG19 for ImageNet and Resnet34 for Caltech-101, respectively, using only $20$ raindrops.



## **43. Efficient Biologically Plausible Adversarial Training**

cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2309.17348v3) [paper-pdf](http://arxiv.org/pdf/2309.17348v3)

**Authors**: Matilde Tristany Farinha, Thomas Ortner, Giorgia Dellaferrera, Benjamin Grewe, Angeliki Pantazi

**Abstract**: Artificial Neural Networks (ANNs) trained with Backpropagation (BP) show astounding performance and are increasingly often used in performing our daily life tasks. However, ANNs are highly vulnerable to adversarial attacks, which alter inputs with small targeted perturbations that drastically disrupt the models' performance. The most effective method to make ANNs robust against these attacks is adversarial training, in which the training dataset is augmented with exemplary adversarial samples. Unfortunately, this approach has the drawback of increased training complexity since generating adversarial samples is very computationally demanding. In contrast to ANNs, humans are not susceptible to adversarial attacks. Therefore, in this work, we investigate whether biologically-plausible learning algorithms are more robust against adversarial attacks than BP. In particular, we present an extensive comparative analysis of the adversarial robustness of BP and Present the Error to Perturb the Input To modulate Activity (PEPITA), a recently proposed biologically-plausible learning algorithm, on various computer vision tasks. We observe that PEPITA has higher intrinsic adversarial robustness and, with adversarial training, has a more favourable natural-vs-adversarial performance trade-off as, for the same natural accuracies, PEPITA's adversarial accuracies decrease in average by 0.26% and BP's by 8.05%.



## **44. An Integrated Algorithm for Robust and Imperceptible Audio Adversarial Examples**

cs.SD

Proc. 3rd Symposium on Security and Privacy in Speech Communication

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03349v1) [paper-pdf](http://arxiv.org/pdf/2310.03349v1)

**Authors**: Armin Ettenhofer, Jan-Philipp Schulze, Karla Pizzi

**Abstract**: Audio adversarial examples are audio files that have been manipulated to fool an automatic speech recognition (ASR) system, while still sounding benign to a human listener. Most methods to generate such samples are based on a two-step algorithm: first, a viable adversarial audio file is produced, then, this is fine-tuned with respect to perceptibility and robustness. In this work, we present an integrated algorithm that uses psychoacoustic models and room impulse responses (RIR) in the generation step. The RIRs are dynamically created by a neural network during the generation process to simulate a physical environment to harden our examples against transformations experienced in over-the-air attacks. We compare the different approaches in three experiments: in a simulated environment and in a realistic over-the-air scenario to evaluate the robustness, and in a human study to evaluate the perceptibility. Our algorithms considering psychoacoustics only or in addition to the robustness show an improvement in the signal-to-noise ratio (SNR) as well as in the human perception study, at the cost of an increased word error rate (WER).



## **45. Certifiably Robust Graph Contrastive Learning**

cs.CR

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03312v1) [paper-pdf](http://arxiv.org/pdf/2310.03312v1)

**Authors**: Minhua Lin, Teng Xiao, Enyan Dai, Xiang Zhang, Suhang Wang

**Abstract**: Graph Contrastive Learning (GCL) has emerged as a popular unsupervised graph representation learning method. However, it has been shown that GCL is vulnerable to adversarial attacks on both the graph structure and node attributes. Although empirical approaches have been proposed to enhance the robustness of GCL, the certifiable robustness of GCL is still remain unexplored. In this paper, we develop the first certifiably robust framework in GCL. Specifically, we first propose a unified criteria to evaluate and certify the robustness of GCL. We then introduce a novel technique, RES (Randomized Edgedrop Smoothing), to ensure certifiable robustness for any GCL model, and this certified robustness can be provably preserved in downstream tasks. Furthermore, an effective training method is proposed for robust GCL. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed method in providing effective certifiable robustness and enhancing the robustness of any GCL model. The source code of RES is available at https://github.com/ventr1c/RES-GCL.



## **46. BaDExpert: Extracting Backdoor Functionality for Accurate Backdoor Input Detection**

cs.CR

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2308.12439v2) [paper-pdf](http://arxiv.org/pdf/2308.12439v2)

**Authors**: Tinghao Xie, Xiangyu Qi, Ping He, Yiming Li, Jiachen T. Wang, Prateek Mittal

**Abstract**: We present a novel defense, against backdoor attacks on Deep Neural Networks (DNNs), wherein adversaries covertly implant malicious behaviors (backdoors) into DNNs. Our defense falls within the category of post-development defenses that operate independently of how the model was generated. The proposed defense is built upon a novel reverse engineering approach that can directly extract backdoor functionality of a given backdoored model to a backdoor expert model. The approach is straightforward -- finetuning the backdoored model over a small set of intentionally mislabeled clean samples, such that it unlearns the normal functionality while still preserving the backdoor functionality, and thus resulting in a model (dubbed a backdoor expert model) that can only recognize backdoor inputs. Based on the extracted backdoor expert model, we show the feasibility of devising highly accurate backdoor input detectors that filter out the backdoor inputs during model inference. Further augmented by an ensemble strategy with a finetuned auxiliary model, our defense, BaDExpert (Backdoor Input Detection with Backdoor Expert), effectively mitigates 17 SOTA backdoor attacks while minimally impacting clean utility. The effectiveness of BaDExpert has been verified on multiple datasets (CIFAR10, GTSRB and ImageNet) across various model architectures (ResNet, VGG, MobileNetV2 and Vision Transformer).



## **47. Burning the Adversarial Bridges: Robust Windows Malware Detection Against Binary-level Mutations**

cs.LG

12 pages

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03285v1) [paper-pdf](http://arxiv.org/pdf/2310.03285v1)

**Authors**: Ahmed Abusnaina, Yizhen Wang, Sunpreet Arora, Ke Wang, Mihai Christodorescu, David Mohaisen

**Abstract**: Toward robust malware detection, we explore the attack surface of existing malware detection systems. We conduct root-cause analyses of the practical binary-level black-box adversarial malware examples. Additionally, we uncover the sensitivity of volatile features within the detection engines and exhibit their exploitability. Highlighting volatile information channels within the software, we introduce three software pre-processing steps to eliminate the attack surface, namely, padding removal, software stripping, and inter-section information resetting. Further, to counter the emerging section injection attacks, we propose a graph-based section-dependent information extraction scheme for software representation. The proposed scheme leverages aggregated information within various sections in the software to enable robust malware detection and mitigate adversarial settings. Our experimental results show that traditional malware detection models are ineffective against adversarial threats. However, the attack surface can be largely reduced by eliminating the volatile information. Therefore, we propose simple-yet-effective methods to mitigate the impacts of binary manipulation attacks. Overall, our graph-based malware detection scheme can accurately detect malware with an area under the curve score of 88.32\% and a score of 88.19% under a combination of binary manipulation attacks, exhibiting the efficiency of our proposed scheme.



## **48. Network Cascade Vulnerability using Constrained Bayesian Optimization**

cs.SI

13 pages, 5 figures

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2304.14420v2) [paper-pdf](http://arxiv.org/pdf/2304.14420v2)

**Authors**: Albert Lam, Mihai Anitescu, Anirudh Subramanyam

**Abstract**: Measures of power grid vulnerability are often assessed by the amount of damage an adversary can exact on the network. However, the cascading impact of such attacks is often overlooked, even though cascades are one of the primary causes of large-scale blackouts. This paper explores modifications of transmission line protection settings as candidates for adversarial attacks, which can remain undetectable as long as the network equilibrium state remains unaltered. This forms the basis of a black-box function in a Bayesian optimization procedure, where the objective is to find protection settings that maximize network degradation due to cascading. Notably, our proposed method is agnostic to the choice of the cascade simulator and its underlying assumptions. Numerical experiments reveal that, against conventional wisdom, maximally misconfiguring the protection settings of all network lines does not cause the most cascading. More surprisingly, even when the degree of misconfiguration is limited due to resource constraints, it is still possible to find settings that produce cascades comparable in severity to instances where there are no resource constraints.



## **49. LinGCN: Structural Linearized Graph Convolutional Network for Homomorphically Encrypted Inference**

cs.LG

NeurIPS 2023 accepted publication

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2309.14331v3) [paper-pdf](http://arxiv.org/pdf/2309.14331v3)

**Authors**: Hongwu Peng, Ran Ran, Yukui Luo, Jiahui Zhao, Shaoyi Huang, Kiran Thorat, Tong Geng, Chenghong Wang, Xiaolin Xu, Wujie Wen, Caiwen Ding

**Abstract**: The growth of Graph Convolution Network (GCN) model sizes has revolutionized numerous applications, surpassing human performance in areas such as personal healthcare and financial systems. The deployment of GCNs in the cloud raises privacy concerns due to potential adversarial attacks on client data. To address security concerns, Privacy-Preserving Machine Learning (PPML) using Homomorphic Encryption (HE) secures sensitive client data. However, it introduces substantial computational overhead in practical applications. To tackle those challenges, we present LinGCN, a framework designed to reduce multiplication depth and optimize the performance of HE based GCN inference. LinGCN is structured around three key elements: (1) A differentiable structural linearization algorithm, complemented by a parameterized discrete indicator function, co-trained with model weights to meet the optimization goal. This strategy promotes fine-grained node-level non-linear location selection, resulting in a model with minimized multiplication depth. (2) A compact node-wise polynomial replacement policy with a second-order trainable activation function, steered towards superior convergence by a two-level distillation approach from an all-ReLU based teacher model. (3) an enhanced HE solution that enables finer-grained operator fusion for node-wise activation functions, further reducing multiplication level consumption in HE-based inference. Our experiments on the NTU-XVIEW skeleton joint dataset reveal that LinGCN excels in latency, accuracy, and scalability for homomorphically encrypted inference, outperforming solutions such as CryptoGCN. Remarkably, LinGCN achieves a 14.2x latency speedup relative to CryptoGCN, while preserving an inference accuracy of 75% and notably reducing multiplication depth.



## **50. Misusing Tools in Large Language Models With Visual Adversarial Examples**

cs.CR

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.03185v1) [paper-pdf](http://arxiv.org/pdf/2310.03185v1)

**Authors**: Xiaohan Fu, Zihan Wang, Shuheng Li, Rajesh K. Gupta, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Models (LLMs) are being enhanced with the ability to use tools and to process multiple modalities. These new capabilities bring new benefits and also new security risks. In this work, we show that an attacker can use visual adversarial examples to cause attacker-desired tool usage. For example, the attacker could cause a victim LLM to delete calendar events, leak private conversations and book hotels. Different from prior work, our attacks can affect the confidentiality and integrity of user resources connected to the LLM while being stealthy and generalizable to multiple input prompts. We construct these attacks using gradient-based adversarial training and characterize performance along multiple dimensions. We find that our adversarial images can manipulate the LLM to invoke tools following real-world syntax almost always (~98%) while maintaining high similarity to clean images (~0.9 SSIM). Furthermore, using human scoring and automated metrics, we find that the attacks do not noticeably affect the conversation (and its semantics) between the user and the LLM.



