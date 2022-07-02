# Latest Adversarial Attack Papers
**update at 2022-07-03 06:31:26**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. MEAD: A Multi-Armed Approach for Evaluation of Adversarial Examples Detectors**

cs.CV

This paper has been accepted to appear in the Proceedings of the 2022  European Conference on Machine Learning and Data Mining (ECML-PKDD), 19th to  the 23rd of September, Grenoble, France

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.15415v1)

**Authors**: Federica Granese, Marine Picot, Marco Romanelli, Francisco Messina, Pablo Piantanida

**Abstracts**: Detection of adversarial examples has been a hot topic in the last years due to its importance for safely deploying machine learning algorithms in critical applications. However, the detection methods are generally validated by assuming a single implicitly known attack strategy, which does not necessarily account for real-life threats. Indeed, this can lead to an overoptimistic assessment of the detectors' performance and may induce some bias in the comparison between competing detection schemes. We propose a novel multi-armed framework, called MEAD, for evaluating detectors based on several attack strategies to overcome this limitation. Among them, we make use of three new objectives to generate attacks. The proposed performance metric is based on the worst-case scenario: detection is successful if and only if all different attacks are correctly recognized. Empirically, we show the effectiveness of our approach. Moreover, the poor performance obtained for state-of-the-art detectors opens a new exciting line of research.



## **2. The Topological BERT: Transforming Attention into Topology for Natural Language Processing**

cs.CL

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.15195v1)

**Authors**: Ilan Perez, Raphael Reinauer

**Abstracts**: In recent years, the introduction of the Transformer models sparked a revolution in natural language processing (NLP). BERT was one of the first text encoders using only the attention mechanism without any recurrent parts to achieve state-of-the-art results on many NLP tasks.   This paper introduces a text classifier using topological data analysis. We use BERT's attention maps transformed into attention graphs as the only input to that classifier. The model can solve tasks such as distinguishing spam from ham messages, recognizing whether a sentence is grammatically correct, or evaluating a movie review as negative or positive. It performs comparably to the BERT baseline and outperforms it on some tasks.   Additionally, we propose a new method to reduce the number of BERT's attention heads considered by the topological classifier, which allows us to prune the number of heads from 144 down to as few as ten with no reduction in performance. Our work also shows that the topological model displays higher robustness against adversarial attacks than the original BERT model, which is maintained during the pruning process. To the best of our knowledge, this work is the first to confront topological-based models with adversarial attacks in the context of NLP.



## **3. FIDO2 With Two Displays$\unicode{x2013}$Or How to Protect Security-Critical Web Transactions Against Malware Attacks**

cs.CR

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.13358v2)

**Authors**: Timon Hackenjos, Benedikt Wagner, Julian Herr, Jochen Rill, Marek Wehmer, Niklas Goerke, Ingmar Baumgart

**Abstracts**: With the rise of attacks on online accounts in the past years, more and more services offer two-factor authentication for their users. Having factors out of two of the three categories something you know, something you have and something you are should ensure that an attacker cannot compromise two of them at once. Thus, an adversary should not be able to maliciously interact with one's account. However, this is only true if one considers a weak adversary. In particular, since most current solutions only authenticate a session and not individual transactions, they are noneffective if one's device is infected with malware. For online banking, the banking industry has long since identified the need for authenticating transactions. However, specifications of such authentication schemes are not public and implementation details vary wildly from bank to bank with most still being unable to protect against malware. In this work, we present a generic approach to tackle the problem of malicious account takeovers, even in the presence of malware. To this end, we define a new paradigm to improve two-factor authentication that involves the concepts of one-out-of-two security and transaction authentication. Web authentication schemes following this paradigm can protect security-critical transactions against manipulation, even if one of the factors is completely compromised. Analyzing existing authentication schemes, we find that they do not realize one-out-of-two security. We give a blueprint of how to design secure web authentication schemes in general. Based on this blueprint we propose FIDO2 With Two Displays (FIDO2D), a new web authentication scheme based on the FIDO2 standard and prove its security using Tamarin. We hope that our work inspires a new wave of more secure web authentication schemes, which protect security-critical transactions even against attacks with malware.



## **4. An Intermediate-level Attack Framework on The Basis of Linear Regression**

cs.CV

Accepted by TPAMI; Code is available at  https://github.com/qizhangli/ila-plus-plus-lr

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2203.10723v2)

**Authors**: Yiwen Guo, Qizhang Li, Wangmeng Zuo, Hao Chen

**Abstracts**: This paper substantially extends our work published at ECCV, in which an intermediate-level attack was proposed to improve the transferability of some baseline adversarial examples. Specifically, we advocate a framework in which a direct linear mapping from the intermediate-level discrepancies (between adversarial features and benign features) to prediction loss of the adversarial example is established. By delving deep into the core components of such a framework, we show that 1) a variety of linear regression models can all be considered in order to establish the mapping, 2) the magnitude of the finally obtained intermediate-level adversarial discrepancy is correlated with the transferability, 3) further boost of the performance can be achieved by performing multiple runs of the baseline attack with random initialization. In addition, by leveraging these findings, we achieve new state-of-the-arts on transfer-based $\ell_\infty$ and $\ell_2$ attacks. Our code is publicly available at https://github.com/qizhangli/ila-plus-plus-lr.



## **5. On the Challenges of Detecting Side-Channel Attacks in SGX**

cs.CR

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2011.14599v2)

**Authors**: Jianyu Jiang, Claudio Soriente, Ghassan Karame

**Abstracts**: Existing tools to detect side-channel attacks on Intel SGX are grounded on the observation that attacks affect the performance of the victim application. As such, all detection tools monitor the potential victim and raise an alarm if the witnessed performance (in terms of runtime, enclave interruptions, cache misses, etc.) is out of the ordinary.   In this paper, we show that monitoring the performance of enclaves to detect side-channel attacks may not be effective. Our core intuition is that all monitoring tools are geared towards an adversary that interferes with the victim's execution in order to extract the most number of secret bits (e.g., the entire secret) in one or few runs. They cannot, however, detect an adversary that leaks smaller portions of the secret - as small as a single bit - at each execution of the victim. In particular, by minimizing the information leaked at each run, the impact of any side-channel attack on the application's performance is significantly lowered - ensuring that the detection tool does not detect an attack. By repeating the attack multiple times, each time on a different part of the secret, the adversary can recover the whole secret and remain undetected. Based on this intuition, we adapt known attacks leveraging page-tables and L3 cache to bypass existing detection mechanisms. We show experimentally how an attacker can successfully exfiltrate the secret key used in an enclave running various cryptographic routines of libgcrypt. Beyond cryptographic libraries, we also show how to compromise the predictions of enclaves running decision-tree routines of OpenCV. Our evaluation results suggest that performance-based detection tools do not deter side-channel attacks on SGX enclaves and that effective detection mechanisms are yet to be designed.



## **6. Depth-2 Neural Networks Under a Data-Poisoning Attack**

cs.LG

32 page, 7 figures

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2005.01699v3)

**Authors**: Sayar Karmakar, Anirbit Mukherjee, Theodore Papamarkou

**Abstracts**: In this work, we study the possibility of defending against data-poisoning attacks while training a shallow neural network in a regression setup. We focus on doing supervised learning for a class of depth-2 finite-width neural networks, which includes single-filter convolutional networks. In this class of networks, we attempt to learn the network weights in the presence of a malicious oracle doing stochastic, bounded and additive adversarial distortions on the true output during training. For the non-gradient stochastic algorithm that we construct, we prove worst-case near-optimal trade-offs among the magnitude of the adversarial attack, the weight approximation accuracy, and the confidence achieved by the proposed algorithm. As our algorithm uses mini-batching, we analyze how the mini-batch size affects convergence. We also show how to utilize the scaling of the outer layer weights to counter output-poisoning attacks depending on the probability of attack. Lastly, we give experimental evidence demonstrating how our algorithm outperforms stochastic gradient descent under different input data distributions, including instances of heavy-tailed distributions.



## **7. IBP Regularization for Verified Adversarial Robustness via Branch-and-Bound**

cs.LG

ICML 2022 Workshop on Formal Verification of Machine Learning

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14772v1)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth

**Abstracts**: Recent works have tried to increase the verifiability of adversarially trained networks by running the attacks over domains larger than the original perturbations and adding various regularization terms to the objective. However, these algorithms either underperform or require complex and expensive stage-wise training procedures, hindering their practical applicability. We present IBP-R, a novel verified training algorithm that is both simple and effective. IBP-R induces network verifiability by coupling adversarial attacks on enlarged domains with a regularization term, based on inexpensive interval bound propagation, that minimizes the gap between the non-convex verification problem and its approximations. By leveraging recent branch-and-bound frameworks, we show that IBP-R obtains state-of-the-art verified robustness-accuracy trade-offs for small perturbations on CIFAR-10 while training significantly faster than relevant previous work. Additionally, we present UPB, a novel branching strategy that, relying on a simple heuristic based on $\beta$-CROWN, reduces the cost of state-of-the-art branching algorithms while yielding splits of comparable quality.



## **8. longhorns at DADC 2022: How many linguists does it take to fool a Question Answering model? A systematic approach to adversarial attacks**

cs.CL

Accepted at DADC2022

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14729v1)

**Authors**: Venelin Kovatchev, Trina Chatterjee, Venkata S Govindarajan, Jifan Chen, Eunsol Choi, Gabriella Chronis, Anubrata Das, Katrin Erk, Matthew Lease, Junyi Jessy Li, Yating Wu, Kyle Mahowald

**Abstracts**: Developing methods to adversarially challenge NLP systems is a promising avenue for improving both model performance and interpretability. Here, we describe the approach of the team "longhorns" on Task 1 of the The First Workshop on Dynamic Adversarial Data Collection (DADC), which asked teams to manually fool a model on an Extractive Question Answering task. Our team finished first, with a model error rate of 62%. We advocate for a systematic, linguistically informed approach to formulating adversarial questions, and we describe the results of our pilot experiments, as well as our official submission.



## **9. Private Graph Extraction via Feature Explanations**

cs.LG

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14724v1)

**Authors**: Iyiola E. Olatunji, Mandeep Rathee, Thorben Funke, Megha Khosla

**Abstracts**: Privacy and interpretability are two of the important ingredients for achieving trustworthy machine learning. We study the interplay of these two aspects in graph machine learning through graph reconstruction attacks. The goal of the adversary here is to reconstruct the graph structure of the training data given access to model explanations. Based on the different kinds of auxiliary information available to the adversary, we propose several graph reconstruction attacks. We show that additional knowledge of post-hoc feature explanations substantially increases the success rate of these attacks. Further, we investigate in detail the differences between attack performance with respect to three different classes of explanation methods for graph neural networks: gradient-based, perturbation-based, and surrogate model-based methods. While gradient-based explanations reveal the most in terms of the graph structure, we find that these explanations do not always score high in utility. For the other two classes of explanations, privacy leakage increases with an increase in explanation utility. Finally, we propose a defense based on a randomized response mechanism for releasing the explanations which substantially reduces the attack success rate. Our anonymized code is available.



## **10. Enhancing Security of Memristor Computing System Through Secure Weight Mapping**

cs.ET

6 pages, 4 figures, accepted by IEEE ISVLSI 2022

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14498v1)

**Authors**: Minhui Zou, Junlong Zhou, Xiaotong Cui, Wei Wang, Shahar Kvatinsky

**Abstracts**: Emerging memristor computing systems have demonstrated great promise in improving the energy efficiency of neural network (NN) algorithms. The NN weights stored in memristor crossbars, however, may face potential theft attacks due to the nonvolatility of the memristor devices. In this paper, we propose to protect the NN weights by mapping selected columns of them in the form of 1's complements and leaving the other columns in their original form, preventing the adversary from knowing the exact representation of each weight. The results show that compared with prior work, our method achieves effectiveness comparable to the best of them and reduces the hardware overhead by more than 18X.



## **11. Adversarial Ensemble Training by Jointly Learning Label Dependencies and Member Models**

cs.LG

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14477v1)

**Authors**: Lele Wang, Bin Liu

**Abstracts**: Training an ensemble of different sub-models has empirically proven to be an effective strategy to improve deep neural networks' adversarial robustness. Current ensemble training methods for image recognition usually encode the image labels by one-hot vectors, which neglect dependency relationships between the labels. Here we propose a novel adversarial training approach that learns the conditional dependencies between labels and the model ensemble jointly. We test our approach on widely used datasets MNIST, FasionMNIST and CIFAR-10. Results show that our approach is more robust against black-box attacks compared with state-of-the-art methods. Our code is available at https://github.com/ZJLAB-AMMI/LSD.



## **12. Guided Diffusion Model for Adversarial Purification**

cs.CV

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2205.14969v3)

**Authors**: Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, Hongfei Fu

**Abstracts**: With wider application of deep neural networks (DNNs) in various algorithms and frameworks, security threats have become one of the concerns. Adversarial attacks disturb DNN-based image classifiers, in which attackers can intentionally add imperceptible adversarial perturbations on input images to fool the classifiers. In this paper, we propose a novel purification approach, referred to as guided diffusion model for purification (GDMP), to help protect classifiers from adversarial attacks. The core of our approach is to embed purification into the diffusion denoising process of a Denoised Diffusion Probabilistic Model (DDPM), so that its diffusion process could submerge the adversarial perturbations with gradually added Gaussian noises, and both of these noises can be simultaneously removed following a guided denoising process. On our comprehensive experiments across various datasets, the proposed GDMP is shown to reduce the perturbations raised by adversarial attacks to a shallow range, thereby significantly improving the correctness of classification. GDMP improves the robust accuracy by 5%, obtaining 90.1% under PGD attack on the CIFAR10 dataset. Moreover, GDMP achieves 70.94% robustness on the challenging ImageNet dataset.



## **13. A Deep Learning Approach to Create DNS Amplification Attacks**

cs.CR

12 pages, 6 figures, Conference: to 2022 4th International Conference  on Management Science and Industrial Engineering (MSIE) (MSIE 2022), DOI:  https://doi.org/10.1145/3535782.3535838, accepted to conference above, not  yet published

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14346v1)

**Authors**: Jared Mathews, Prosenjit Chatterjee, Shankar Banik, Cory Nance

**Abstracts**: In recent years, deep learning has shown itself to be an incredibly valuable tool in cybersecurity as it helps network intrusion detection systems to classify attacks and detect new ones. Adversarial learning is the process of utilizing machine learning to generate a perturbed set of inputs to then feed to the neural network to misclassify it. Much of the current work in the field of adversarial learning has been conducted in image processing and natural language processing with a wide variety of algorithms. Two algorithms of interest are the Elastic-Net Attack on Deep Neural Networks and TextAttack. In our experiment the EAD and TextAttack algorithms are applied to a Domain Name System amplification classifier. The algorithms are used to generate malicious Distributed Denial of Service adversarial examples to then feed as inputs to the network intrusion detection systems neural network to classify as valid traffic. We show in this work that both image processing and natural language processing adversarial learning algorithms can be applied against a network intrusion detection neural network.



## **14. Linear Model Against Malicious Adversaries with Local Differential Privacy**

cs.CR

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2202.02448v2)

**Authors**: Guanhong Miao, A. Adam Ding, Samuel S. Wu

**Abstracts**: Scientific collaborations benefit from collaborative learning of distributed sources, but remain difficult to achieve when data are sensitive. In recent years, privacy preserving techniques have been widely studied to analyze distributed data across different agencies while protecting sensitive information. Most existing privacy preserving techniques are designed to resist semi-honest adversaries and require intense computation to perform data analysis. Secure collaborative learning is significantly difficult with the presence of malicious adversaries who may deviates from the secure protocol. Another challenge is to maintain high computation efficiency with privacy protection. In this paper, matrix encryption is applied to encrypt data such that the secure schemes are against malicious adversaries, including chosen plaintext attack, known plaintext attack, and collusion attack. The encryption scheme also achieves local differential privacy. Moreover, cross validation is studied to prevent overfitting without additional communication cost. Empirical experiments on real-world datasets demonstrate that the proposed schemes are computationally efficient compared to existing techniques against malicious adversary and semi-honest model.



## **15. An Empirical Study of Challenges in Converting Deep Learning Models**

cs.LG

Accepted for publication in ICSME 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14322v1)

**Authors**: Moses Openja, Amin Nikanjam, Ahmed Haj Yahmed, Foutse Khomh, Zhen Ming, Jiang

**Abstracts**: There is an increase in deploying Deep Learning (DL)-based software systems in real-world applications. Usually DL models are developed and trained using DL frameworks that have their own internal mechanisms/formats to represent and train DL models, and usually those formats cannot be recognized by other frameworks. Moreover, trained models are usually deployed in environments different from where they were developed. To solve the interoperability issue and make DL models compatible with different frameworks/environments, some exchange formats are introduced for DL models, like ONNX and CoreML. However, ONNX and CoreML were never empirically evaluated by the community to reveal their prediction accuracy, performance, and robustness after conversion. Poor accuracy or non-robust behavior of converted models may lead to poor quality of deployed DL-based software systems. We conduct, in this paper, the first empirical study to assess ONNX and CoreML for converting trained DL models. In our systematic approach, two popular DL frameworks, Keras and PyTorch, are used to train five widely used DL models on three popular datasets. The trained models are then converted to ONNX and CoreML and transferred to two runtime environments designated for such formats, to be evaluated. We investigate the prediction accuracy before and after conversion. Our results unveil that the prediction accuracy of converted models are at the same level of originals. The performance (time cost and memory consumption) of converted models are studied as well. The size of models are reduced after conversion, which can result in optimized DL-based software deployment. Converted models are generally assessed as robust at the same level of originals. However, obtained results show that CoreML models are more vulnerable to adversarial attacks compared to ONNX.



## **16. Collecting high-quality adversarial data for machine reading comprehension tasks with humans and models in the loop**

cs.CL

8 pages, 3 figures, for more information about the shared task please  go to https://dadcworkshop.github.io/

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14272v1)

**Authors**: Damian Y. Romero Diaz, Magdalena Anioł, John Culnan

**Abstracts**: We present our experience as annotators in the creation of high-quality, adversarial machine-reading-comprehension data for extractive QA for Task 1 of the First Workshop on Dynamic Adversarial Data Collection (DADC). DADC is an emergent data collection paradigm with both models and humans in the loop. We set up a quasi-experimental annotation design and perform quantitative analyses across groups with different numbers of annotators focusing on successful adversarial attacks, cost analysis, and annotator confidence correlation. We further perform a qualitative analysis of our perceived difficulty of the task given the different topics of the passages in our dataset and conclude with recommendations and suggestions that might be of value to people working on future DADC tasks and related annotation interfaces.



## **17. How to Steer Your Adversary: Targeted and Efficient Model Stealing Defenses with Gradient Redirection**

cs.LG

ICML 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14157v1)

**Authors**: Mantas Mazeika, Bo Li, David Forsyth

**Abstracts**: Model stealing attacks present a dilemma for public machine learning APIs. To protect financial investments, companies may be forced to withhold important information about their models that could facilitate theft, including uncertainty estimates and prediction explanations. This compromise is harmful not only to users but also to external transparency. Model stealing defenses seek to resolve this dilemma by making models harder to steal while preserving utility for benign users. However, existing defenses have poor performance in practice, either requiring enormous computational overheads or severe utility trade-offs. To meet these challenges, we present a new approach to model stealing defenses called gradient redirection. At the core of our approach is a provably optimal, efficient algorithm for steering an adversary's training updates in a targeted manner. Combined with improvements to surrogate networks and a novel coordinated defense strategy, our gradient redirection defense, called GRAD${}^2$, achieves small utility trade-offs and low computational overhead, outperforming the best prior defenses. Moreover, we demonstrate how gradient redirection enables reprogramming the adversary with arbitrary behavior, which we hope will foster work on new avenues of defense.



## **18. Debiasing Learning for Membership Inference Attacks Against Recommender Systems**

cs.IR

Accepted by KDD 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.12401v2)

**Authors**: Zihan Wang, Na Huang, Fei Sun, Pengjie Ren, Zhumin Chen, Hengliang Luo, Maarten de Rijke, Zhaochun Ren

**Abstracts**: Learned recommender systems may inadvertently leak information about their training data, leading to privacy violations. We investigate privacy threats faced by recommender systems through the lens of membership inference. In such attacks, an adversary aims to infer whether a user's data is used to train the target recommender. To achieve this, previous work has used a shadow recommender to derive training data for the attack model, and then predicts the membership by calculating difference vectors between users' historical interactions and recommended items. State-of-the-art methods face two challenging problems: (1) training data for the attack model is biased due to the gap between shadow and target recommenders, and (2) hidden states in recommenders are not observational, resulting in inaccurate estimations of difference vectors. To address the above limitations, we propose a Debiasing Learning for Membership Inference Attacks against recommender systems (DL-MIA) framework that has four main components: (1) a difference vector generator, (2) a disentangled encoder, (3) a weight estimator, and (4) an attack model. To mitigate the gap between recommenders, a variational auto-encoder (VAE) based disentangled encoder is devised to identify recommender invariant and specific features. To reduce the estimation bias, we design a weight estimator, assigning a truth-level score for each difference vector to indicate estimation accuracy. We evaluate DL-MIA against both general recommenders and sequential recommenders on three real-world datasets. Experimental results show that DL-MIA effectively alleviates training and estimation biases simultaneously, and achieves state-of-the-art attack performance.



## **19. On the amplification of security and privacy risks by post-hoc explanations in machine learning models**

cs.LG

9 pages, appendix: 2 pages

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14004v1)

**Authors**: Pengrui Quan, Supriyo Chakraborty, Jeya Vikranth Jeyakumar, Mani Srivastava

**Abstracts**: A variety of explanation methods have been proposed in recent years to help users gain insights into the results returned by neural networks, which are otherwise complex and opaque black-boxes. However, explanations give rise to potential side-channels that can be leveraged by an adversary for mounting attacks on the system. In particular, post-hoc explanation methods that highlight input dimensions according to their importance or relevance to the result also leak information that weakens security and privacy. In this work, we perform the first systematic characterization of the privacy and security risks arising from various popular explanation techniques. First, we propose novel explanation-guided black-box evasion attacks that lead to 10 times reduction in query count for the same success rate. We show that the adversarial advantage from explanations can be quantified as a reduction in the total variance of the estimated gradient. Second, we revisit the membership information leaked by common explanations. Contrary to observations in prior studies, via our modified attacks we show significant leakage of membership information (above 100% improvement over prior results), even in a much stricter black-box setting. Finally, we study explanation-guided model extraction attacks and demonstrate adversarial gains through a large reduction in query count.



## **20. Increasing Confidence in Adversarial Robustness Evaluations**

cs.LG

Oral at CVPR 2022 Workshop (Art of Robustness). Project website  https://zimmerrol.github.io/active-tests/

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.13991v1)

**Authors**: Roland S. Zimmermann, Wieland Brendel, Florian Tramer, Nicholas Carlini

**Abstracts**: Hundreds of defenses have been proposed to make deep neural networks robust against minimal (adversarial) input perturbations. However, only a handful of these defenses held up their claims because correctly evaluating robustness is extremely challenging: Weak attacks often fail to find adversarial examples even if they unknowingly exist, thereby making a vulnerable network look robust. In this paper, we propose a test to identify weak attacks, and thus weak defense evaluations. Our test slightly modifies a neural network to guarantee the existence of an adversarial example for every sample. Consequentially, any correct attack must succeed in breaking this modified network. For eleven out of thirteen previously-published defenses, the original evaluation of the defense fails our test, while stronger attacks that break these defenses pass it. We hope that attack unit tests - such as ours - will be a major component in future robustness evaluations and increase confidence in an empirical field that is currently riddled with skepticism.



## **21. Ownership Verification of DNN Architectures via Hardware Cache Side Channels**

cs.CR

The paper has been accepted by IEEE Transactions on Circuits and  Systems for Video Technology

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2102.03523v4)

**Authors**: Xiaoxuan Lou, Shangwei Guo, Jiwei Li, Tianwei Zhang

**Abstracts**: Deep Neural Networks (DNN) are gaining higher commercial values in computer vision applications, e.g., image classification, video analytics, etc. This calls for urgent demands of the intellectual property (IP) protection of DNN models. In this paper, we present a novel watermarking scheme to achieve the ownership verification of DNN architectures. Existing works all embedded watermarks into the model parameters while treating the architecture as public property. These solutions were proven to be vulnerable by an adversary to detect or remove the watermarks. In contrast, we claim the model architectures as an important IP for model owners, and propose to implant watermarks into the architectures. We design new algorithms based on Neural Architecture Search (NAS) to generate watermarked architectures, which are unique enough to represent the ownership, while maintaining high model usability. Such watermarks can be extracted via side-channel-based model extraction techniques with high fidelity. We conduct comprehensive experiments on watermarked CNN models for image classification tasks and the experimental results show our scheme has negligible impact on the model performance, and exhibits strong robustness against various model transformations and adaptive attacks.



## **22. Deep Image Destruction: Vulnerability of Deep Image-to-Image Models against Adversarial Attacks**

cs.CV

ICPR2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2104.15022v2)

**Authors**: Jun-Ho Choi, Huan Zhang, Jun-Hyuk Kim, Cho-Jui Hsieh, Jong-Seok Lee

**Abstracts**: Recently, the vulnerability of deep image classification models to adversarial attacks has been investigated. However, such an issue has not been thoroughly studied for image-to-image tasks that take an input image and generate an output image (e.g., colorization, denoising, deblurring, etc.) This paper presents comprehensive investigations into the vulnerability of deep image-to-image models to adversarial attacks. For five popular image-to-image tasks, 16 deep models are analyzed from various standpoints such as output quality degradation due to attacks, transferability of adversarial examples across different tasks, and characteristics of perturbations. We show that unlike image classification tasks, the performance degradation on image-to-image tasks largely differs depending on various factors, e.g., attack methods and task objectives. In addition, we analyze the effectiveness of conventional defense methods used for classification models in improving the robustness of the image-to-image models.



## **23. Improving Privacy and Security in Unmanned Aerial Vehicles Network using Blockchain**

cs.CR

18 Pages; 14 Figures; 2 Tables

**SubmitDate**: 2022-06-27    [paper-pdf](http://arxiv.org/pdf/2201.06100v2)

**Authors**: Hardik Sachdeva, Shivam Gupta, Anushka Misra, Khushbu Chauhan, Mayank Dave

**Abstracts**: Unmanned Aerial Vehicles (UAVs), also known as drones, have exploded in every segment present in todays business industry. They have scope in reinventing old businesses, and they are even developing new opportunities for various brands and franchisors. UAVs are used in the supply chain, maintaining surveillance and serving as mobile hotspots. Although UAVs have potential applications, they bring several societal concerns and challenges that need addressing in public safety, privacy, and cyber security. UAVs are prone to various cyber-attacks and vulnerabilities; they can also be hacked and misused by malicious entities resulting in cyber-crime. The adversaries can exploit these vulnerabilities, leading to data loss, property, and destruction of life. One can partially detect the attacks like false information dissemination, jamming, gray hole, blackhole, and GPS spoofing by monitoring the UAV behavior, but it may not resolve privacy issues. This paper presents secure communication between UAVs using blockchain technology. Our approach involves building smart contracts and making a secure and reliable UAV adhoc network. This network will be resilient to various network attacks and is secure against malicious intrusions.



## **24. Adversarially Robust Learning of Real-Valued Functions**

cs.LG

**SubmitDate**: 2022-06-26    [paper-pdf](http://arxiv.org/pdf/2206.12977v1)

**Authors**: Idan Attias, Steve Hanneke

**Abstracts**: We study robustness to test-time adversarial attacks in the regression setting with $\ell_p$ losses and arbitrary perturbation sets. We address the question of which function classes are PAC learnable in this setting. We show that classes of finite fat-shattering dimension are learnable. Moreover, for convex function classes, they are even properly learnable. In contrast, some non-convex function classes provably require improper learning algorithms. We also discuss extensions to agnostic learning. Our main technique is based on a construction of an adversarially robust sample compression scheme of a size determined by the fat-shattering dimension.



## **25. Cascading Failures in Smart Grids under Random, Targeted and Adaptive Attacks**

cs.SI

Accepted for publication as a book chapter. arXiv admin note:  substantial text overlap with arXiv:1402.6809

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12735v1)

**Authors**: Sushmita Ruj, Arindam Pal

**Abstracts**: We study cascading failures in smart grids, where an attacker selectively compromises the nodes with probabilities proportional to their degrees, betweenness, or clustering coefficient. This implies that nodes with high degrees, betweenness, or clustering coefficients are attacked with higher probability. We mathematically and experimentally analyze the sizes of the giant components of the networks under different types of targeted attacks, and compare the results with the corresponding sizes under random attacks. We show that networks disintegrate faster for targeted attacks compared to random attacks. A targeted attack on a small fraction of high degree nodes disintegrates one or both of the networks, whereas both the networks contain giant components for random attack on the same fraction of nodes. An important observation is that an attacker has an advantage if it compromises nodes based on their betweenness, rather than based on degree or clustering coefficient.   We next study adaptive attacks, where an attacker compromises nodes in rounds. Here, some nodes are compromised in each round based on their degree, betweenness or clustering coefficients, instead of compromising all nodes together. In this case, the degree, betweenness, or clustering coefficient is calculated before the start of each round, instead of at the beginning. We show experimentally that an adversary has an advantage in this adaptive approach, compared to compromising the same number of nodes all at once.



## **26. Empirical Evaluation of Physical Adversarial Patch Attacks Against Overhead Object Detection Models**

cs.CV

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12725v1)

**Authors**: Gavin S. Hartnett, Li Ang Zhang, Caolionn O'Connell, Andrew J. Lohn, Jair Aguirre

**Abstracts**: Adversarial patches are images designed to fool otherwise well-performing neural network-based computer vision models. Although these attacks were initially conceived of and studied digitally, in that the raw pixel values of the image were perturbed, recent work has demonstrated that these attacks can successfully transfer to the physical world. This can be accomplished by printing out the patch and adding it into scenes of newly captured images or video footage. In this work we further test the efficacy of adversarial patch attacks in the physical world under more challenging conditions. We consider object detection models trained on overhead imagery acquired through aerial or satellite cameras, and we test physical adversarial patches inserted into scenes of a desert environment. Our main finding is that it is far more difficult to successfully implement the adversarial patch attacks under these conditions than in the previously considered conditions. This has important implications for AI safety as the real-world threat posed by adversarial examples may be overstated.



## **27. Defending Multimodal Fusion Models against Single-Source Adversaries**

cs.CV

CVPR 2021

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12714v1)

**Authors**: Karren Yang, Wan-Yi Lin, Manash Barman, Filipe Condessa, Zico Kolter

**Abstracts**: Beyond achieving high performance across many vision tasks, multimodal models are expected to be robust to single-source faults due to the availability of redundant information between modalities. In this paper, we investigate the robustness of multimodal neural networks against worst-case (i.e., adversarial) perturbations on a single modality. We first show that standard multimodal fusion models are vulnerable to single-source adversaries: an attack on any single modality can overcome the correct information from multiple unperturbed modalities and cause the model to fail. This surprising vulnerability holds across diverse multimodal tasks and necessitates a solution. Motivated by this finding, we propose an adversarially robust fusion strategy that trains the model to compare information coming from all the input sources, detect inconsistencies in the perturbed modality compared to the other modalities, and only allow information from the unperturbed modalities to pass through. Our approach significantly improves on state-of-the-art methods in single-source robustness, achieving gains of 7.8-25.2% on action recognition, 19.7-48.2% on object detection, and 1.6-6.7% on sentiment analysis, without degrading performance on unperturbed (i.e., clean) data.



## **28. Defense against adversarial attacks on deep convolutional neural networks through nonlocal denoising**

cs.CV

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12685v1)

**Authors**: Sandhya Aneja, Nagender Aneja, Pg Emeroylariffion Abas, Abdul Ghani Naim

**Abstracts**: Despite substantial advances in network architecture performance, the susceptibility of adversarial attacks makes deep learning challenging to implement in safety-critical applications. This paper proposes a data-centric approach to addressing this problem. A nonlocal denoising method with different luminance values has been used to generate adversarial examples from the Modified National Institute of Standards and Technology database (MNIST) and Canadian Institute for Advanced Research (CIFAR-10) data sets. Under perturbation, the method provided absolute accuracy improvements of up to 9.3% in the MNIST data set and 13% in the CIFAR-10 data set. Training using transformed images with higher luminance values increases the robustness of the classifier. We have shown that transfer learning is disadvantageous for adversarial machine learning. The results indicate that simple adversarial examples can improve resilience and make deep learning easier to apply in various applications.



## **29. RSTAM: An Effective Black-Box Impersonation Attack on Face Recognition using a Mobile and Compact Printer**

cs.CV

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12590v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao, Changhai Nie

**Abstracts**: Face recognition has achieved considerable progress in recent years thanks to the development of deep neural networks, but it has recently been discovered that deep neural networks are vulnerable to adversarial examples. This means that face recognition models or systems based on deep neural networks are also susceptible to adversarial examples. However, the existing methods of attacking face recognition models or systems with adversarial examples can effectively complete white-box attacks but not black-box impersonation attacks, physical attacks, or convenient attacks, particularly on commercial face recognition systems. In this paper, we propose a new method to attack face recognition models or systems called RSTAM, which enables an effective black-box impersonation attack using an adversarial mask printed by a mobile and compact printer. First, RSTAM enhances the transferability of the adversarial masks through our proposed random similarity transformation strategy. Furthermore, we propose a random meta-optimization strategy for ensembling several pre-trained face models to generate more general adversarial masks. Finally, we conduct experiments on the CelebA-HQ, LFW, Makeup Transfer (MT), and CASIA-FaceV5 datasets. The performance of the attacks is also evaluated on state-of-the-art commercial face recognition systems: Face++, Baidu, Aliyun, Tencent, and Microsoft. Extensive experiments show that RSTAM can effectively perform black-box impersonation attacks on face recognition models or systems.



## **30. Defending Backdoor Attacks on Vision Transformer via Patch Processing**

cs.CV

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12381v1)

**Authors**: Khoa D. Doan, Yingjie Lao, Peng Yang, Ping Li

**Abstracts**: Vision Transformers (ViTs) have a radically different architecture with significantly less inductive bias than Convolutional Neural Networks. Along with the improvement in performance, security and robustness of ViTs are also of great importance to study. In contrast to many recent works that exploit the robustness of ViTs against adversarial examples, this paper investigates a representative causative attack, i.e., backdoor. We first examine the vulnerability of ViTs against various backdoor attacks and find that ViTs are also quite vulnerable to existing attacks. However, we observe that the clean-data accuracy and backdoor attack success rate of ViTs respond distinctively to patch transformations before the positional encoding. Then, based on this finding, we propose an effective method for ViTs to defend both patch-based and blending-based trigger backdoor attacks via patch processing. The performances are evaluated on several benchmark datasets, including CIFAR10, GTSRB, and TinyImageNet, which show the proposed novel defense is very successful in mitigating backdoor attacks for ViTs. To the best of our knowledge, this paper presents the first defensive strategy that utilizes a unique characteristic of ViTs against backdoor attacks.



## **31. Robustness of Explanation Methods for NLP Models**

cs.CL

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12284v1)

**Authors**: Shriya Atmakuri, Tejas Chheda, Dinesh Kandula, Nishant Yadav, Taesung Lee, Hessel Tuinhof

**Abstracts**: Explanation methods have emerged as an important tool to highlight the features responsible for the predictions of neural networks. There is mounting evidence that many explanation methods are rather unreliable and susceptible to malicious manipulations. In this paper, we particularly aim to understand the robustness of explanation methods in the context of text modality. We provide initial insights and results towards devising a successful adversarial attack against text explanations. To our knowledge, this is the first attempt to evaluate the adversarial robustness of an explanation method. Our experiments show the explanation method can be largely disturbed for up to 86% of the tested samples with small changes in the input sentence and its semantics.



## **32. Property Unlearning: A Defense Strategy Against Property Inference Attacks**

cs.CR

Please note: As of June 24, 2022, we have discovered some flaws in  our experimental setup. The defense mechanism property unlearning is not as  strong as the experimental results in the current version of the paper  suggest. We will provide an updated version soon

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2205.08821v2)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstracts**: During the training of machine learning models, they may store or "learn" more information about the training data than what is actually needed for the prediction or classification task. This is exploited by property inference attacks which aim at extracting statistical properties from the training data of a given model without having access to the training data itself. These properties may include the quality of pictures to identify the camera model, the age distribution to reveal the target audience of a product, or the included host types to refine a malware attack in computer networks. This attack is especially accurate when the attacker has access to all model parameters, i.e., in a white-box scenario. By defending against such attacks, model owners are able to ensure that their training data, associated properties, and thus their intellectual property stays private, even if they deliberately share their models, e.g., to train collaboratively, or if models are leaked. In this paper, we introduce property unlearning, an effective defense mechanism against white-box property inference attacks, independent of the training data type, model task, or number of properties. Property unlearning mitigates property inference attacks by systematically changing the trained weights and biases of a target model such that an adversary cannot extract chosen properties. We empirically evaluate property unlearning on three different data sets, including tabular and image data, and two types of artificial neural networks. Our results show that property unlearning is both efficient and reliable to protect machine learning models against property inference attacks, with a good privacy-utility trade-off. Furthermore, our approach indicates that this mechanism is also effective to unlearn multiple properties.



## **33. Adversarial Robustness of Deep Neural Networks: A Survey from a Formal Verification Perspective**

cs.CR

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12227v1)

**Authors**: Mark Huasong Meng, Guangdong Bai, Sin Gee Teo, Zhe Hou, Yan Xiao, Yun Lin, Jin Song Dong

**Abstracts**: Neural networks have been widely applied in security applications such as spam and phishing detection, intrusion prevention, and malware detection. This black-box method, however, often has uncertainty and poor explainability in applications. Furthermore, neural networks themselves are often vulnerable to adversarial attacks. For those reasons, there is a high demand for trustworthy and rigorous methods to verify the robustness of neural network models. Adversarial robustness, which concerns the reliability of a neural network when dealing with maliciously manipulated inputs, is one of the hottest topics in security and machine learning. In this work, we survey existing literature in adversarial robustness verification for neural networks and collect 39 diversified research works across machine learning, security, and software engineering domains. We systematically analyze their approaches, including how robustness is formulated, what verification techniques are used, and the strengths and limitations of each technique. We provide a taxonomy from a formal verification perspective for a comprehensive understanding of this topic. We classify the existing techniques based on property specification, problem reduction, and reasoning strategies. We also demonstrate representative techniques that have been applied in existing studies with a sample model. Finally, we discuss open questions for future research.



## **34. Cluster Attack: Query-based Adversarial Attacks on Graphs with Graph-Dependent Priors**

cs.LG

IJCAI 2022 (Long Presentation)

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2109.13069v2)

**Authors**: Zhengyi Wang, Zhongkai Hao, Ziqiao Wang, Hang Su, Jun Zhu

**Abstracts**: While deep neural networks have achieved great success in graph analysis, recent work has shown that they are vulnerable to adversarial attacks. Compared with adversarial attacks on image classification, performing adversarial attacks on graphs is more challenging because of the discrete and non-differential nature of the adjacent matrix for a graph. In this work, we propose Cluster Attack -- a Graph Injection Attack (GIA) on node classification, which injects fake nodes into the original graph to degenerate the performance of graph neural networks (GNNs) on certain victim nodes while affecting the other nodes as little as possible. We demonstrate that a GIA problem can be equivalently formulated as a graph clustering problem; thus, the discrete optimization problem of the adjacency matrix can be solved in the context of graph clustering. In particular, we propose to measure the similarity between victim nodes by a metric of Adversarial Vulnerability, which is related to how the victim nodes will be affected by the injected fake node, and to cluster the victim nodes accordingly. Our attack is performed in a practical and unnoticeable query-based black-box manner with only a few nodes on the graphs that can be accessed. Theoretical analysis and extensive experiments demonstrate the effectiveness of our method by fooling the node classifiers with only a small number of queries.



## **35. An Improved Lattice-Based Ring Signature with Unclaimable Anonymity in the Standard Model**

cs.CR

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12093v1)

**Authors**: Mingxing Hu, Weijiong Zhang, Zhen Liu

**Abstracts**: Ring signatures enable a user to sign messages on behalf of an arbitrary set of users, called the ring, without revealing exactly which member of that ring actually generated the signature. The signer-anonymity property makes ring signatures have been an active research topic. Recently, Park and Sealfon (CRYPTO 19) presented an important anonymity notion named signer-unclaimability and constructed a lattice-based ring signature scheme with unclaimable anonymity in the standard model, however, it did not consider the unforgeable w.r.t. adversarially-chosen-key attack (the public key ring of a signature may contain keys created by an adversary) and the signature size grows quadratically in the size of ring and message. In this work, we propose a new lattice-based ring signature scheme with unclaimable anonymity in the standard model. In particular, our work improves the security and efficiency of Park and Sealfons work, which is unforgeable w.r.t. adversarially-chosen-key attack, and the ring signature size grows linearly in the ring size.



## **36. Keep Your Transactions On Short Leashes**

cs.CR

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11974v1)

**Authors**: Bennet Yee

**Abstracts**: The adversary's goal in mounting Long Range Attacks (LRAs) is to fool potential victims into using and relying on a side chain, i.e., a false, alternate history of transactions, and into proposing transactions that end up harming themselves or others. Previous research work on LRAs on blockchain systems have used, at a high level, one of two approaches. They either try to (1) prevent the creation of a bogus side chain or (2) make it possible to distinguish such a side chain from the main consensus chain.   In this paper, we take a different approach. We start with the indistinguishability of side chains from the consensus chain -- for the eclipsed victim -- as a given and assume the potential victim will be fooled. Instead, we protect the victim via harm reduction applying "short leashes" to transactions. The leashes prevent transactions from being used in the wrong context.   The primary contribution of this paper is the design and analysis of leashes. A secondary contribution is the careful explication of the LRA threat model in the context of BAR fault tolerance, and using it to analyze related work to identify their limitations.



## **37. Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attacks**

cs.CR

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2108.05075v3)

**Authors**: Zitao Chen, Pritam Dash, Karthik Pattabiraman

**Abstracts**: Adversarial patch attacks that inject arbitrary distortions within a bounded region of an image, can trigger misclassification in deep neural networks (DNNs). These attacks are robust (i.e., physically realizable) and universally malicious, and hence represent a severe security threat to real-world DNN-based systems.   This work proposes Jujutsu, a two-stage technique to detect and mitigate robust and universal adversarial patch attacks. We first observe that patch attacks often yield large influence on the prediction output in order to dominate the prediction on any input, and Jujutsu is built to expose this behavior for effective attack detection. For mitigation, we observe that patch attacks corrupt only a localized region while the remaining contents are unperturbed, based on which Jujutsu leverages GAN-based image inpainting to synthesize the semantic contents in the pixels that are corrupted by the attacks, and reconstruct the ``clean'' image for correct prediction.   We evaluate Jujutsu on four diverse datasets and show that it achieves superior performance and significantly outperforms four leading defenses. Jujutsu can further defend against physical-world attacks, attacks that target diverse classes, and adaptive attacks. Our code is available at https://github.com/DependableSystemsLab/Jujutsu.



## **38. Probabilistically Resilient Multi-Robot Informative Path Planning**

cs.RO

9 pages, 6 figures, submitted to IEEE Robotics and Automation Letters  (RA-L)

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11789v1)

**Authors**: Remy Wehbe, Ryan K. Williams

**Abstracts**: In this paper, we solve a multi-robot informative path planning (MIPP) task under the influence of uncertain communication and adversarial attackers. The goal is to create a multi-robot system that can learn and unify its knowledge of an unknown environment despite the presence of corrupted robots sharing malicious information. We use a Gaussian Process (GP) to model our unknown environment and define informativeness using the metric of mutual information. The objectives of our MIPP task is to maximize the amount of information collected by the team while maximizing the probability of resilience to attack. Unfortunately, these objectives are at odds especially when exploring large environments which necessitates disconnections between robots. As a result, we impose a probabilistic communication constraint that allows robots to meet intermittently and resiliently share information, and then act to maximize collected information during all other times. To solve our problem, we select meeting locations with the highest probability of resilience and use a sequential greedy algorithm to optimize paths for robots to explore. Finally, we show the validity of our results by comparing the learning ability of well-behaving robots applying resilient vs. non-resilient MIPP algorithms.



## **39. Towards End-to-End Private Automatic Speaker Recognition**

eess.AS

Accepted for publication at Interspeech 2022

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11750v1)

**Authors**: Francisco Teixeira, Alberto Abad, Bhiksha Raj, Isabel Trancoso

**Abstracts**: The development of privacy-preserving automatic speaker verification systems has been the focus of a number of studies with the intent of allowing users to authenticate themselves without risking the privacy of their voice. However, current privacy-preserving methods assume that the template voice representations (or speaker embeddings) used for authentication are extracted locally by the user. This poses two important issues: first, knowledge of the speaker embedding extraction model may create security and robustness liabilities for the authentication system, as this knowledge might help attackers in crafting adversarial examples able to mislead the system; second, from the point of view of a service provider the speaker embedding extraction model is arguably one of the most valuable components in the system and, as such, disclosing it would be highly undesirable. In this work, we show how speaker embeddings can be extracted while keeping both the speaker's voice and the service provider's model private, using Secure Multiparty Computation. Further, we show that it is possible to obtain reasonable trade-offs between security and computational cost. This work is complementary to those showing how authentication may be performed privately, and thus can be considered as another step towards fully private automatic speaker recognition.



## **40. BERT Rankers are Brittle: a Study using Adversarial Document Perturbations**

cs.IR

To appear in ICTIR 2022

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11724v1)

**Authors**: Yumeng Wang, Lijun Lyu, Avishek Anand

**Abstracts**: Contextual ranking models based on BERT are now well established for a wide range of passage and document ranking tasks. However, the robustness of BERT-based ranking models under adversarial inputs is under-explored. In this paper, we argue that BERT-rankers are not immune to adversarial attacks targeting retrieved documents given a query. Firstly, we propose algorithms for adversarial perturbation of both highly relevant and non-relevant documents using gradient-based optimization methods. The aim of our algorithms is to add/replace a small number of tokens to a highly relevant or non-relevant document to cause a large rank demotion or promotion. Our experiments show that a small number of tokens can already result in a large change in the rank of a document. Moreover, we find that BERT-rankers heavily rely on the document start/head for relevance prediction, making the initial part of the document more susceptible to adversarial attacks. More interestingly, we find a small set of recurring adversarial words that when added to documents result in successful rank demotion/promotion of any relevant/non-relevant document respectively. Finally, our adversarial tokens also show particular topic preferences within and across datasets, exposing potential biases from BERT pre-training or downstream datasets.



## **41. Adversarial Zoom Lens: A Novel Physical-World Attack to DNNs**

cs.CR

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.12251v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Although deep neural networks (DNNs) are known to be fragile, no one has studied the effects of zooming-in and zooming-out of images in the physical world on DNNs performance. In this paper, we demonstrate a novel physical adversarial attack technique called Adversarial Zoom Lens (AdvZL), which uses a zoom lens to zoom in and out of pictures of the physical world, fooling DNNs without changing the characteristics of the target object. The proposed method is so far the only adversarial attack technique that does not add physical adversarial perturbation attack DNNs. In a digital environment, we construct a data set based on AdvZL to verify the antagonism of equal-scale enlarged images to DNNs. In the physical environment, we manipulate the zoom lens to zoom in and out of the target object, and generate adversarial samples. The experimental results demonstrate the effectiveness of AdvZL in both digital and physical environments. We further analyze the antagonism of the proposed data set to the improved DNNs. On the other hand, we provide a guideline for defense against AdvZL by means of adversarial training. Finally, we look into the threat possibilities of the proposed approach to future autonomous driving and variant attack ideas similar to the proposed attack.



## **42. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

cs.CV

6 pages workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.06761v2)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.



## **43. Bounding Training Data Reconstruction in Private (Deep) Learning**

cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2201.12383v4)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.



## **44. A Framework for Understanding Model Extraction Attack and Defense**

cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11480v1)

**Authors**: Xun Xian, Mingyi Hong, Jie Ding

**Abstracts**: The privacy of machine learning models has become a significant concern in many emerging Machine-Learning-as-a-Service applications, where prediction services based on well-trained models are offered to users via pay-per-query. The lack of a defense mechanism can impose a high risk on the privacy of the server's model since an adversary could efficiently steal the model by querying only a few `good' data points. The interplay between a server's defense and an adversary's attack inevitably leads to an arms race dilemma, as commonly seen in Adversarial Machine Learning. To study the fundamental tradeoffs between model utility from a benign user's view and privacy from an adversary's view, we develop new metrics to quantify such tradeoffs, analyze their theoretical properties, and develop an optimization problem to understand the optimal adversarial attack and defense strategies. The developed concepts and theory match the empirical findings on the `equilibrium' between privacy and utility. In terms of optimization, the key ingredient that enables our results is a unified representation of the attack-defense problem as a min-max bi-level problem. The developed results will be demonstrated by examples and experiments.



## **45. InfoAT: Improving Adversarial Training Using the Information Bottleneck Principle**

cs.LG

Published in: IEEE Transactions on Neural Networks and Learning  Systems ( Early Access )

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.12292v1)

**Authors**: Mengting Xu, Tao Zhang, Zhongnian Li, Daoqiang Zhang

**Abstracts**: Adversarial training (AT) has shown excellent high performance in defending against adversarial examples. Recent studies demonstrate that examples are not equally important to the final robustness of models during AT, that is, the so-called hard examples that can be attacked easily exhibit more influence than robust examples on the final robustness. Therefore, guaranteeing the robustness of hard examples is crucial for improving the final robustness of the model. However, defining effective heuristics to search for hard examples is still difficult. In this article, inspired by the information bottleneck (IB) principle, we uncover that an example with high mutual information of the input and its associated latent representation is more likely to be attacked. Based on this observation, we propose a novel and effective adversarial training method (InfoAT). InfoAT is encouraged to find examples with high mutual information and exploit them efficiently to improve the final robustness of models. Experimental results show that InfoAT achieves the best robustness among different datasets and models in comparison with several state-of-the-art methods.



## **46. Incorporating Hidden Layer representation into Adversarial Attacks and Defences**

cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2011.14045v2)

**Authors**: Haojing Shen, Sihong Chen, Ran Wang, Xizhao Wang

**Abstracts**: In this paper, we propose a defence strategy to improve adversarial robustness by incorporating hidden layer representation. The key of this defence strategy aims to compress or filter input information including adversarial perturbation. And this defence strategy can be regarded as an activation function which can be applied to any kind of neural network. We also prove theoretically the effectiveness of this defense strategy under certain conditions. Besides, incorporating hidden layer representation we propose three types of adversarial attacks to generate three types of adversarial examples, respectively. The experiments show that our defence method can significantly improve the adversarial robustness of deep neural networks which achieves the state-of-the-art performance even though we do not adopt adversarial training.



## **47. Adversarial Learning with Cost-Sensitive Classes**

cs.LG

12 pages

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2101.12372v2)

**Authors**: Haojing Shen, Sihong Chen, Ran Wang, Xizhao Wang

**Abstracts**: It is necessary to improve the performance of some special classes or to particularly protect them from attacks in adversarial learning. This paper proposes a framework combining cost-sensitive classification and adversarial learning together to train a model that can distinguish between protected and unprotected classes, such that the protected classes are less vulnerable to adversarial examples. We find in this framework an interesting phenomenon during the training of deep neural networks, called Min-Max property, that is, the absolute values of most parameters in the convolutional layer approach zero while the absolute values of a few parameters are significantly larger becoming bigger. Based on this Min-Max property which is formulated and analyzed in a view of random distribution, we further build a new defense model against adversarial examples for adversarial robustness improvement. An advantage of the built model is that it performs better than the standard one and can combine with adversarial training to achieve an improved performance. It is experimentally confirmed that, regarding the average accuracy of all classes, our model is almost as same as the existing models when an attack does not occur and is better than the existing models when an attack occurs. Specifically, regarding the accuracy of protected classes, the proposed model is much better than the existing models when an attack occurs.



## **48. Shilling Black-box Recommender Systems by Learning to Generate Fake User Profiles**

cs.IR

Accepted by TNNLS. 15 pages, 8 figures

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11433v1)

**Authors**: Chen Lin, Si Chen, Meifang Zeng, Sheng Zhang, Min Gao, Hui Li

**Abstracts**: Due to the pivotal role of Recommender Systems (RS) in guiding customers towards the purchase, there is a natural motivation for unscrupulous parties to spoof RS for profits. In this paper, we study Shilling Attack where an adversarial party injects a number of fake user profiles for improper purposes. Conventional Shilling Attack approaches lack attack transferability (i.e., attacks are not effective on some victim RS models) and/or attack invisibility (i.e., injected profiles can be easily detected). To overcome these issues, we present Leg-UP, a novel attack model based on the Generative Adversarial Network. Leg-UP learns user behavior patterns from real users in the sampled ``templates'' and constructs fake user profiles. To simulate real users, the generator in Leg-UP directly outputs discrete ratings. To enhance attack transferability, the parameters of the generator are optimized by maximizing the attack performance on a surrogate RS model. To improve attack invisibility, Leg-UP adopts a discriminator to guide the generator to generate undetectable fake user profiles. Experiments on benchmarks have shown that Leg-UP exceeds state-of-the-art Shilling Attack methods on a wide range of victim RS models. The source code of our work is available at: https://github.com/XMUDM/ShillingAttack.



## **49. Making Generated Images Hard To Spot: A Transferable Attack On Synthetic Image Detectors**

cs.CV

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2104.12069v2)

**Authors**: Xinwei Zhao, Matthew C. Stamm

**Abstracts**: Visually realistic GAN-generated images have recently emerged as an important misinformation threat. Research has shown that these synthetic images contain forensic traces that are readily identifiable by forensic detectors. Unfortunately, these detectors are built upon neural networks, which are vulnerable to recently developed adversarial attacks. In this paper, we propose a new anti-forensic attack capable of fooling GAN-generated image detectors. Our attack uses an adversarially trained generator to synthesize traces that these detectors associate with real images. Furthermore, we propose a technique to train our attack so that it can achieve transferability, i.e. it can fool unknown CNNs that it was not explicitly trained against. We evaluate our attack through an extensive set of experiments, where we show that our attack can fool eight state-of-the-art detection CNNs with synthetic images created using seven different GANs, and outperform other alternative attacks.



## **50. AdvSmo: Black-box Adversarial Attack by Smoothing Linear Structure of Texture**

cs.CV

6 pages,3 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10988v1)

**Authors**: Hui Xia, Rui Zhang, Shuliang Jiang, Zi Kang

**Abstracts**: Black-box attacks usually face two problems: poor transferability and the inability to evade the adversarial defense. To overcome these shortcomings, we create an original approach to generate adversarial examples by smoothing the linear structure of the texture in the benign image, called AdvSmo. We construct the adversarial examples without relying on any internal information to the target model and design the imperceptible-high attack success rate constraint to guide the Gabor filter to select appropriate angles and scales to smooth the linear texture from the input images to generate adversarial examples. Benefiting from the above design concept, AdvSmo will generate adversarial examples with strong transferability and solid evasiveness. Finally, compared to the four advanced black-box adversarial attack methods, for the eight target models, the results show that AdvSmo improves the average attack success rate by 9% on the CIFAR-10 and 16% on the Tiny-ImageNet dataset compared to the best of these attack methods.



