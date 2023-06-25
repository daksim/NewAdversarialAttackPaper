# Latest Adversarial Attack Papers
**update at 2023-06-25 16:52:04**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Evading Forensic Classifiers with Attribute-Conditioned Adversarial Faces**

cs.CV

Accepted in CVPR 2023. Project page:  https://koushiksrivats.github.io/face_attribute_attack/

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13091v1) [paper-pdf](http://arxiv.org/pdf/2306.13091v1)

**Authors**: Fahad Shamshad, Koushik Srivatsan, Karthik Nandakumar

**Abstract**: The ability of generative models to produce highly realistic synthetic face images has raised security and ethical concerns. As a first line of defense against such fake faces, deep learning based forensic classifiers have been developed. While these forensic models can detect whether a face image is synthetic or real with high accuracy, they are also vulnerable to adversarial attacks. Although such attacks can be highly successful in evading detection by forensic classifiers, they introduce visible noise patterns that are detectable through careful human scrutiny. Additionally, these attacks assume access to the target model(s) which may not always be true. Attempts have been made to directly perturb the latent space of GANs to produce adversarial fake faces that can circumvent forensic classifiers. In this work, we go one step further and show that it is possible to successfully generate adversarial fake faces with a specified set of attributes (e.g., hair color, eye size, race, gender, etc.). To achieve this goal, we leverage the state-of-the-art generative model StyleGAN with disentangled representations, which enables a range of modifications without leaving the manifold of natural images. We propose a framework to search for adversarial latent codes within the feature space of StyleGAN, where the search can be guided either by a text prompt or a reference image. We also propose a meta-learning based optimization strategy to achieve transferable performance on unknown target models. Extensive experiments demonstrate that the proposed approach can produce semantically manipulated adversarial fake faces, which are true to the specified attribute set and can successfully fool forensic face classifiers, while remaining undetectable by humans. Code: https://github.com/koushiksrivats/face_attribute_attack.



## **2. Impacts and Risk of Generative AI Technology on Cyber Defense**

cs.CR

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13033v1) [paper-pdf](http://arxiv.org/pdf/2306.13033v1)

**Authors**: Subash Neupane, Ivan A. Fernandez, Sudip Mittal, Shahram Rahimi

**Abstract**: Generative Artificial Intelligence (GenAI) has emerged as a powerful technology capable of autonomously producing highly realistic content in various domains, such as text, images, audio, and videos. With its potential for positive applications in creative arts, content generation, virtual assistants, and data synthesis, GenAI has garnered significant attention and adoption. However, the increasing adoption of GenAI raises concerns about its potential misuse for crafting convincing phishing emails, generating disinformation through deepfake videos, and spreading misinformation via authentic-looking social media posts, posing a new set of challenges and risks in the realm of cybersecurity. To combat the threats posed by GenAI, we propose leveraging the Cyber Kill Chain (CKC) to understand the lifecycle of cyberattacks, as a foundational model for cyber defense. This paper aims to provide a comprehensive analysis of the risk areas introduced by the offensive use of GenAI techniques in each phase of the CKC framework. We also analyze the strategies employed by threat actors and examine their utilization throughout different phases of the CKC, highlighting the implications for cyber defense. Additionally, we propose GenAI-enabled defense strategies that are both attack-aware and adaptive. These strategies encompass various techniques such as detection, deception, and adversarial training, among others, aiming to effectively mitigate the risks posed by GenAI-induced cyber threats.



## **3. AI Security for Geoscience and Remote Sensing: Challenges and Future Trends**

cs.CV

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2212.09360v2) [paper-pdf](http://arxiv.org/pdf/2212.09360v2)

**Authors**: Yonghao Xu, Tao Bai, Weikang Yu, Shizhen Chang, Peter M. Atkinson, Pedram Ghamisi

**Abstract**: Recent advances in artificial intelligence (AI) have significantly intensified research in the geoscience and remote sensing (RS) field. AI algorithms, especially deep learning-based ones, have been developed and applied widely to RS data analysis. The successful application of AI covers almost all aspects of Earth observation (EO) missions, from low-level vision tasks like super-resolution, denoising and inpainting, to high-level vision tasks like scene classification, object detection and semantic segmentation. While AI techniques enable researchers to observe and understand the Earth more accurately, the vulnerability and uncertainty of AI models deserve further attention, considering that many geoscience and RS tasks are highly safety-critical. This paper reviews the current development of AI security in the geoscience and RS field, covering the following five important aspects: adversarial attack, backdoor attack, federated learning, uncertainty and explainability. Moreover, the potential opportunities and trends are discussed to provide insights for future research. To the best of the authors' knowledge, this paper is the first attempt to provide a systematic review of AI security-related research in the geoscience and RS community. Available code and datasets are also listed in the paper to move this vibrant field of research forward.



## **4. Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models**

cs.CV

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12941v1) [paper-pdf](http://arxiv.org/pdf/2306.12941v1)

**Authors**: Francesco Croce, Naman D Singh, Matthias Hein

**Abstract**: While a large amount of work has focused on designing adversarial attacks against image classifiers, only a few methods exist to attack semantic segmentation models. We show that attacking segmentation models presents task-specific challenges, for which we propose novel solutions. Our final evaluation protocol outperforms existing methods, and shows that those can overestimate the robustness of the models. Additionally, so far adversarial training, the most successful way for obtaining robust image classifiers, could not be successfully applied to semantic segmentation. We argue that this is because the task to be learned is more challenging, and requires significantly higher computational effort than for image classification. As a remedy, we show that by taking advantage of recent advances in robust ImageNet classifiers, one can train adversarially robust segmentation models at limited computational cost by fine-tuning robust backbones.



## **5. Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation**

cs.CL

Work in progress

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12916v1) [paper-pdf](http://arxiv.org/pdf/2306.12916v1)

**Authors**: Ran Zhang, Jihed Ouni, Steffen Eger

**Abstract**: While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility, information sharing, and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We build the first CLCTS corpus, leveraging historical fictive texts and Wikipedia summaries in English and German, and examine the effectiveness of popular transformer end-to-end models with different intermediate task finetuning tasks. Additionally, we explore the potential of ChatGPT for CLCTS as a summarizer and an evaluator. Overall, we report evaluations from humans, ChatGPT, and several recent automatic evaluation metrics where we find our intermediate task finetuned end-to-end models generate bad to moderate quality summaries; ChatGPT as a summarizer (without any finetuning) provides moderate to good quality outputs and as an evaluator correlates moderately with human evaluations though it is prone to giving lower scores. ChatGPT also seems to be very adept at normalizing historical text. We finally test ChatGPT in a scenario with adversarially attacked and unseen source documents and find that ChatGPT is better at omission and entity swap than negating against its prior knowledge.



## **6. On the explainable properties of 1-Lipschitz Neural Networks: An Optimal Transport Perspective**

cs.AI

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2206.06854v2) [paper-pdf](http://arxiv.org/pdf/2206.06854v2)

**Authors**: Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Béthune, Thibaut Boissin

**Abstract**: Input gradients have a pivotal role in a variety of applications, including adversarial attack algorithms for evaluating model robustness, explainable AI techniques for generating Saliency Maps, and counterfactual explanations. However, Saliency Maps generated by traditional neural networks are often noisy and provide limited insights. In this paper, we demonstrate that, on the contrary, the Saliency Maps of 1-Lipschitz neural networks, learnt with the dual loss of an optimal transportation problem, exhibit desirable XAI properties: They are highly concentrated on the essential parts of the image with low noise, significantly outperforming state-of-the-art explanation approaches across various models and metrics. We also prove that these maps align unprecedentedly well with human explanations on ImageNet. To explain the particularly beneficial properties of the Saliency Map for such models, we prove this gradient encodes both the direction of the transportation plan and the direction towards the nearest adversarial attack. Following the gradient down to the decision boundary is no longer considered an adversarial attack, but rather a counterfactual explanation that explicitly transports the input from one class to another. Thus, Learning with such a loss jointly optimizes the classification objective and the alignment of the gradient , i.e. the Saliency Map, to the transportation plan direction. These networks were previously known to be certifiably robust by design, and we demonstrate that they scale well for large problems and models, and are tailored for explainability using a fast and straightforward method.



## **7. Conditional Generators for Limit Order Book Environments: Explainability, Challenges, and Robustness**

q-fin.TR

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12806v1) [paper-pdf](http://arxiv.org/pdf/2306.12806v1)

**Authors**: Andrea Coletta, Joseph Jerome, Rahul Savani, Svitlana Vyetrenko

**Abstract**: Limit order books are a fundamental and widespread market mechanism. This paper investigates the use of conditional generative models for order book simulation. For developing a trading agent, this approach has drawn recent attention as an alternative to traditional backtesting due to its ability to react to the presence of the trading agent. Using a state-of-the-art CGAN (from Coletta et al. (2022)), we explore its dependence upon input features, which highlights both strengths and weaknesses. To do this, we use "adversarial attacks" on the model's features and its mechanism. We then show how these insights can be used to improve the CGAN, both in terms of its realism and robustness. We finish by laying out a roadmap for future work.



## **8. Towards quantum enhanced adversarial robustness in machine learning**

quant-ph

10 Pages, 4 Figures

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12688v1) [paper-pdf](http://arxiv.org/pdf/2306.12688v1)

**Authors**: Maxwell T. West, Shu-Lok Tsang, Jia S. Low, Charles D. Hill, Christopher Leckie, Lloyd C. L. Hollenberg, Sarah M. Erfani, Muhammad Usman

**Abstract**: Machine learning algorithms are powerful tools for data driven tasks such as image classification and feature detection, however their vulnerability to adversarial examples - input samples manipulated to fool the algorithm - remains a serious challenge. The integration of machine learning with quantum computing has the potential to yield tools offering not only better accuracy and computational efficiency, but also superior robustness against adversarial attacks. Indeed, recent work has employed quantum mechanical phenomena to defend against adversarial attacks, spurring the rapid development of the field of quantum adversarial machine learning (QAML) and potentially yielding a new source of quantum advantage. Despite promising early results, there remain challenges towards building robust real-world QAML tools. In this review we discuss recent progress in QAML and identify key challenges. We also suggest future research directions which could determine the route to practicality for QAML approaches as quantum computing hardware scales up and noise levels are reduced.



## **9. On the Security Risks of Knowledge Graph Reasoning**

cs.CR

In proceedings of USENIX Security'23. Codes:  https://github.com/HarrialX/security-risk-KG-reasoning

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2305.02383v2) [paper-pdf](http://arxiv.org/pdf/2305.02383v2)

**Authors**: Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Xiapu Luo, Xusheng Xiao, Fenglong Ma, Ting Wang

**Abstract**: Knowledge graph reasoning (KGR) -- answering complex logical queries over large knowledge graphs -- represents an important artificial intelligence task, entailing a range of applications (e.g., cyber threat hunting). However, despite its surging popularity, the potential security risks of KGR are largely unexplored, which is concerning, given the increasing use of such capability in security-critical domains.   This work represents a solid initial step towards bridging the striking gap. We systematize the security threats to KGR according to the adversary's objectives, knowledge, and attack vectors. Further, we present ROAR, a new class of attacks that instantiate a variety of such threats. Through empirical evaluation in representative use cases (e.g., medical decision support, cyber threat hunting, and commonsense reasoning), we demonstrate that ROAR is highly effective to mislead KGR to suggest pre-defined answers for target queries, yet with negligible impact on non-target ones. Finally, we explore potential countermeasures against ROAR, including filtering of potentially poisoning knowledge and training with adversarially augmented queries, which leads to several promising research directions.



## **10. Rethinking the Backward Propagation for Adversarial Transferability**

cs.CV

14 pages

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12685v1) [paper-pdf](http://arxiv.org/pdf/2306.12685v1)

**Authors**: Xiaosen Wang, Kangheng Tong, Kun He

**Abstract**: Transfer-based attacks generate adversarial examples on the surrogate model, which can mislead other black-box models without any access, making it promising to attack real-world applications. Recently, several works have been proposed to boost adversarial transferability, in which the surrogate model is usually overlooked. In this work, we identify that non-linear layers (e.g., ReLU, max-pooling, etc.) truncate the gradient during backward propagation, making the gradient w.r.t.input image imprecise to the loss function. We hypothesize and empirically validate that such truncation undermines the transferability of adversarial examples. Based on these findings, we propose a novel method called Backward Propagation Attack (BPA) to increase the relevance between the gradient w.r.t. input image and loss function so as to generate adversarial examples with higher transferability. Specifically, BPA adopts a non-monotonic function as the derivative of ReLU and incorporates softmax with temperature to smooth the derivative of max-pooling, thereby mitigating the information loss during the backward propagation of gradients. Empirical results on the ImageNet dataset demonstrate that not only does our method substantially boost the adversarial transferability, but it also is general to existing transfer-based attacks.



## **11. FDINet: Protecting against DNN Model Extraction via Feature Distortion Index**

cs.CR

13 pages, 7 figures

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.11338v2) [paper-pdf](http://arxiv.org/pdf/2306.11338v2)

**Authors**: Hongwei Yao, Zheng Li, Haiqin Weng, Feng Xue, Kui Ren, Zhan Qin

**Abstract**: Machine Learning as a Service (MLaaS) platforms have gained popularity due to their accessibility, cost-efficiency, scalability, and rapid development capabilities. However, recent research has highlighted the vulnerability of cloud-based models in MLaaS to model extraction attacks. In this paper, we introduce FDINET, a novel defense mechanism that leverages the feature distribution of deep neural network (DNN) models. Concretely, by analyzing the feature distribution from the adversary's queries, we reveal that the feature distribution of these queries deviates from that of the model's training set. Based on this key observation, we propose Feature Distortion Index (FDI), a metric designed to quantitatively measure the feature distribution deviation of received queries. The proposed FDINET utilizes FDI to train a binary detector and exploits FDI similarity to identify colluding adversaries from distributed extraction attacks. We conduct extensive experiments to evaluate FDINET against six state-of-the-art extraction attacks on four benchmark datasets and four popular model architectures. Empirical results demonstrate the following findings FDINET proves to be highly effective in detecting model extraction, achieving a 100% detection accuracy on DFME and DaST. FDINET is highly efficient, using just 50 queries to raise an extraction alarm with an average confidence of 96.08% for GTSRB. FDINET exhibits the capability to identify colluding adversaries with an accuracy exceeding 91%. Additionally, it demonstrates the ability to detect two types of adaptive attacks.



## **12. SNAP: Efficient Extraction of Private Properties with Poisoning**

cs.LG

28 pages, 16 figures

**SubmitDate**: 2023-06-21    [abs](http://arxiv.org/abs/2208.12348v2) [paper-pdf](http://arxiv.org/pdf/2208.12348v2)

**Authors**: Harsh Chaudhari, John Abascal, Alina Oprea, Matthew Jagielski, Florian Tramèr, Jonathan Ullman

**Abstract**: Property inference attacks allow an adversary to extract global properties of the training dataset from a machine learning model. Such attacks have privacy implications for data owners sharing their datasets to train machine learning models. Several existing approaches for property inference attacks against deep neural networks have been proposed, but they all rely on the attacker training a large number of shadow models, which induces a large computational overhead.   In this paper, we consider the setting of property inference attacks in which the attacker can poison a subset of the training dataset and query the trained target model. Motivated by our theoretical analysis of model confidences under poisoning, we design an efficient property inference attack, SNAP, which obtains higher attack success and requires lower amounts of poisoning than the state-of-the-art poisoning-based property inference attack by Mahloujifar et al. For example, on the Census dataset, SNAP achieves 34% higher success rate than Mahloujifar et al. while being 56.5x faster. We also extend our attack to infer whether a certain property was present at all during training and estimate the exact proportion of a property of interest efficiently. We evaluate our attack on several properties of varying proportions from four datasets and demonstrate SNAP's generality and effectiveness. An open-source implementation of SNAP can be found at https://github.com/johnmath/snap-sp23.



## **13. Adversarial Attacks Neutralization via Data Set Randomization**

cs.LG

**SubmitDate**: 2023-06-21    [abs](http://arxiv.org/abs/2306.12161v1) [paper-pdf](http://arxiv.org/pdf/2306.12161v1)

**Authors**: Mouna Rabhi, Roberto Di Pietro

**Abstract**: Adversarial attacks on deep-learning models pose a serious threat to their reliability and security. Existing defense mechanisms are narrow addressing a specific type of attack or being vulnerable to sophisticated attacks. We propose a new defense mechanism that, while being focused on image-based classifiers, is general with respect to the cited category. It is rooted on hyperspace projection. In particular, our solution provides a pseudo-random projection of the original dataset into a new dataset. The proposed defense mechanism creates a set of diverse projected datasets, where each projected dataset is used to train a specific classifier, resulting in different trained classifiers with different decision boundaries. During testing, it randomly selects a classifier to test the input. Our approach does not sacrifice accuracy over legitimate input. Other than detailing and providing a thorough characterization of our defense mechanism, we also provide a proof of concept of using four optimization-based adversarial attacks (PGD, FGSM, IGSM, and C\&W) and a generative adversarial attack testing them on the MNIST dataset. Our experimental results show that our solution increases the robustness of deep learning models against adversarial attacks and significantly reduces the attack success rate by at least 89% for optimization attacks and 78% for generative attacks. We also analyze the relationship between the number of used hyperspaces and the efficacy of the defense mechanism. As expected, the two are positively correlated, offering an easy-to-tune parameter to enforce the desired level of security. The generality and scalability of our solution and adaptability to different attack scenarios, combined with the excellent achieved results, other than providing a robust defense against adversarial attacks on deep learning networks, also lay the groundwork for future research in the field.



## **14. Sample Attackability in Natural Language Adversarial Attacks**

cs.CL

**SubmitDate**: 2023-06-21    [abs](http://arxiv.org/abs/2306.12043v1) [paper-pdf](http://arxiv.org/pdf/2306.12043v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Adversarial attack research in natural language processing (NLP) has made significant progress in designing powerful attack methods and defence approaches. However, few efforts have sought to identify which source samples are the most attackable or robust, i.e. can we determine for an unseen target model, which samples are the most vulnerable to an adversarial attack. This work formally extends the definition of sample attackability/robustness for NLP attacks. Experiments on two popular NLP datasets, four state of the art models and four different NLP adversarial attack methods, demonstrate that sample uncertainty is insufficient for describing characteristics of attackable/robust samples and hence a deep learning based detector can perform much better at identifying the most attackable and robust samples for an unseen target model. Nevertheless, further analysis finds that there is little agreement in which samples are considered the most attackable/robust across different NLP attack methods, explaining a lack of portability of attackability detection methods across attack methods.



## **15. Evaluating Adversarial Robustness of Convolution-based Human Motion Prediction**

cs.CV

**SubmitDate**: 2023-06-21    [abs](http://arxiv.org/abs/2306.11990v1) [paper-pdf](http://arxiv.org/pdf/2306.11990v1)

**Authors**: Chengxu Duan, Zhicheng Zhang, Xiaoli Liu, Yonghao Dang, Jianqin Yin

**Abstract**: Human motion prediction has achieved a brilliant performance with the help of CNNs, which facilitates human-machine cooperation. However, currently, there is no work evaluating the potential risk in human motion prediction when facing adversarial attacks, which may cause danger in real applications. The adversarial attack will face two problems against human motion prediction: 1. For naturalness, pose data is highly related to the physical dynamics of human skeletons where Lp norm constraints cannot constrain the adversarial example well; 2. Unlike the pixel value in images, pose data is diverse at scale because of the different acquisition equipment and the data processing, which makes it hard to set fixed parameters to perform attacks. To solve the problems above, we propose a new adversarial attack method that perturbs the input human motion sequence by maximizing the prediction error with physical constraints. Specifically, we introduce a novel adaptable scheme that facilitates the attack to suit the scale of the target pose and two physical constraints to enhance the imperceptibility of the adversarial example. The evaluating experiments on three datasets show that the prediction errors of all target models are enlarged significantly, which means current convolution-based human motion prediction models can be easily disturbed under the proposed attack. The quantitative analysis shows that prior knowledge and semantic information modeling can be the key to the adversarial robustness of human motion predictors. The qualitative results indicate that the adversarial sample is hard to be noticed when compared frame by frame but is relatively easy to be detected when the sample is animated.



## **16. Universal adversarial perturbations for multiple classification tasks with quantum classifiers**

quant-ph

**SubmitDate**: 2023-06-21    [abs](http://arxiv.org/abs/2306.11974v1) [paper-pdf](http://arxiv.org/pdf/2306.11974v1)

**Authors**: Yun-Zhong Qiu

**Abstract**: Quantum adversarial machine learning is an emerging field that studies the vulnerability of quantum learning systems against adversarial perturbations and develops possible defense strategies. Quantum universal adversarial perturbations are small perturbations, which can make different input samples into adversarial examples that may deceive a given quantum classifier. This is a field that was rarely looked into but worthwhile investigating because universal perturbations might simplify malicious attacks to a large extent, causing unexpected devastation to quantum machine learning models. In this paper, we take a step forward and explore the quantum universal perturbations in the context of heterogeneous classification tasks. In particular, we find that quantum classifiers that achieve almost state-of-the-art accuracy on two different classification tasks can be both conclusively deceived by one carefully-crafted universal perturbation. This result is explicitly demonstrated with well-designed quantum continual learning models with elastic weight consolidation method to avoid catastrophic forgetting, as well as real-life heterogeneous datasets from hand-written digits and medical MRI images. Our results provide a simple and efficient way to generate universal perturbations on heterogeneous classification tasks and thus would provide valuable guidance for future quantum learning technologies.



## **17. Spectral Augmentation for Self-Supervised Learning on Graphs**

cs.LG

ICLR 2023

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2210.00643v2) [paper-pdf](http://arxiv.org/pdf/2210.00643v2)

**Authors**: Lu Lin, Jinghui Chen, Hongning Wang

**Abstract**: Graph contrastive learning (GCL), as an emerging self-supervised learning technique on graphs, aims to learn representations via instance discrimination. Its performance heavily relies on graph augmentation to reflect invariant patterns that are robust to small perturbations; yet it still remains unclear about what graph invariance GCL should capture. Recent studies mainly perform topology augmentations in a uniformly random manner in the spatial domain, ignoring its influence on the intrinsic structural properties embedded in the spectral domain. In this work, we aim to find a principled way for topology augmentations by exploring the invariance of graphs from the spectral perspective. We develop spectral augmentation which guides topology augmentations by maximizing the spectral change. Extensive experiments on both graph and node classification tasks demonstrate the effectiveness of our method in self-supervised representation learning. The proposed method also brings promising generalization capability in transfer learning, and is equipped with intriguing robustness property under adversarial attacks. Our study sheds light on a general principle for graph topology augmentation.



## **18. Towards a robust and reliable deep learning approach for detection of compact binary mergers in gravitational wave data**

gr-qc

22 pages, 21 figures

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2306.11797v1) [paper-pdf](http://arxiv.org/pdf/2306.11797v1)

**Authors**: Shreejit Jadhav, Mihir Shrivastava, Sanjit Mitra

**Abstract**: The ability of deep learning (DL) approaches to learn generalised signal and noise models, coupled with their fast inference on GPUs, holds great promise for enhancing gravitational-wave (GW) searches in terms of speed, parameter space coverage, and search sensitivity. However, the opaque nature of DL models severely harms their reliability. In this work, we meticulously develop a DL model stage-wise and work towards improving its robustness and reliability. First, we address the problems in maintaining the purity of training data by deriving a new metric that better reflects the visual strength of the "chirp" signal features in the data. Using a reduced, smooth representation obtained through a variational auto-encoder (VAE), we build a classifier to search for compact binary coalescence (CBC) signals. Our tests on real LIGO data show an impressive performance of the model. However, upon probing the robustness of the model through adversarial attacks, its simple failure modes were identified, underlining how such models can still be highly fragile. As a first step towards bringing robustness, we retrain the model in a novel framework involving a generative adversarial network (GAN). Over the course of training, the model learns to eliminate the primary modes of failure identified by the adversaries. Although absolute robustness is practically impossible to achieve, we demonstrate some fundamental improvements earned through such training, like sparseness and reduced degeneracy in the extracted features at different layers inside the model. Through comparative inference on real LIGO data, we show that the prescribed robustness is achieved at practically zero cost in terms of performance. Through a direct search on ~8.8 days of LIGO data, we recover two significant CBC events from GWTC-2.1, GW190519_153544 and GW190521_074359, and report the search sensitivity.



## **19. Illusory Attacks: Detectability Matters in Adversarial Attacks on Sequential Decision-Makers**

cs.AI

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2207.10170v3) [paper-pdf](http://arxiv.org/pdf/2207.10170v3)

**Authors**: Tim Franzmeyer, Stephen McAleer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstract**: Autonomous agents deployed in the real world need to be robust against adversarial attacks on sensory inputs. Robustifying agent policies requires anticipating the strongest attacks possible. We demonstrate that existing observation-space attacks on reinforcement learning agents have a common weakness: while effective, their lack of temporal consistency makes them detectable using automated means or human inspection. Detectability is undesirable to adversaries as it may trigger security escalations. We introduce perfect illusory attacks, a novel form of adversarial attack on sequential decision-makers that is both effective and provably statistically undetectable. We then propose the more versatile R-attacks, which result in observation transitions that are consistent with the state-transition function of the adversary-free environment and can be learned end-to-end. Compared to existing attacks, we empirically find R-attacks to be significantly harder to detect with automated methods, and a small study with human subjects suggests they are similarly harder to detect for humans. We propose that undetectability should be a central concern in the study of adversarial attacks on mixed-autonomy settings.



## **20. Robust Adversarial Attacks Detection based on Explainable Deep Reinforcement Learning For UAV Guidance and Planning**

cs.LG

13 pages, 16 figures

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2206.02670v4) [paper-pdf](http://arxiv.org/pdf/2206.02670v4)

**Authors**: Thomas Hickling, Nabil Aouf, Phillippa Spencer

**Abstract**: The dangers of adversarial attacks on Uncrewed Aerial Vehicle (UAV) agents operating in public are increasing. Adopting AI-based techniques and, more specifically, Deep Learning (DL) approaches to control and guide these UAVs can be beneficial in terms of performance but can add concerns regarding the safety of those techniques and their vulnerability against adversarial attacks. Confusion in the agent's decision-making process caused by these attacks can seriously affect the safety of the UAV. This paper proposes an innovative approach based on the explainability of DL methods to build an efficient detector that will protect these DL schemes and the UAVs adopting them from attacks. The agent adopts a Deep Reinforcement Learning (DRL) scheme for guidance and planning. The agent is trained with a Deep Deterministic Policy Gradient (DDPG) with Prioritised Experience Replay (PER) DRL scheme that utilises Artificial Potential Field (APF) to improve training times and obstacle avoidance performance. A simulated environment for UAV explainable DRL-based planning and guidance, including obstacles and adversarial attacks, is built. The adversarial attacks are generated by the Basic Iterative Method (BIM) algorithm and reduced obstacle course completion rates from 97\% to 35\%. Two adversarial attack detectors are proposed to counter this reduction. The first one is a Convolutional Neural Network Adversarial Detector (CNN-AD), which achieves accuracy in the detection of 80\%. The second detector utilises a Long Short Term Memory (LSTM) network. It achieves an accuracy of 91\% with faster computing times compared to the CNN-AD, allowing for real-time adversarial detection.



## **21. Reversible Adversarial Examples with Beam Search Attack and Grayscale Invariance**

cs.CR

Submitted to ICICS2023

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2306.11322v1) [paper-pdf](http://arxiv.org/pdf/2306.11322v1)

**Authors**: Haodong Zhang, Chi Man Pun, Xia Du

**Abstract**: Reversible adversarial examples (RAE) combine adversarial attacks and reversible data-hiding technology on a single image to prevent illegal access. Most RAE studies focus on achieving white-box attacks. In this paper, we propose a novel framework to generate reversible adversarial examples, which combines a novel beam search based black-box attack and reversible data hiding with grayscale invariance (RDH-GI). This RAE uses beam search to evaluate the adversarial gain of historical perturbations and guide adversarial perturbations. After the adversarial examples are generated, the framework RDH-GI embeds the secret data that can be recovered losslessly. Experimental results show that our method can achieve an average Peak Signal-to-Noise Ratio (PSNR) of at least 40dB compared to source images with limited query budgets. Our method can also achieve a targeted black-box reversible adversarial attack for the first time.



## **22. Comparative Evaluation of Recent Universal Adversarial Perturbations in Image Classification**

cs.CV

18 pages,8 figures, 7 tables

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2306.11261v1) [paper-pdf](http://arxiv.org/pdf/2306.11261v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li

**Abstract**: The vulnerability of Convolutional Neural Networks (CNNs) to adversarial samples has recently garnered significant attention in the machine learning community. Furthermore, recent studies have unveiled the existence of universal adversarial perturbations (UAPs) that are image-agnostic and highly transferable across different CNN models. In this survey, our primary focus revolves around the recent advancements in UAPs specifically within the image classification task. We categorize UAPs into two distinct categories, i.e., noise-based attacks and generator-based attacks, thereby providing a comprehensive overview of representative methods within each category. By presenting the computational details of these methods, we summarize various loss functions employed for learning UAPs. Furthermore, we conduct a comprehensive evaluation of different loss functions within consistent training frameworks, including noise-based and generator-based. The evaluation covers a wide range of attack settings, including black-box and white-box attacks, targeted and untargeted attacks, as well as the examination of defense mechanisms.   Our quantitative evaluation results yield several important findings pertaining to the effectiveness of different loss functions, the selection of surrogate CNN models, the impact of training data and data size, and the training frameworks involved in crafting universal attackers. Finally, to further promote future research on universal adversarial attacks, we provide some visualizations of the perturbations and discuss the potential research directions.



## **23. Adversarial Robustness of Learning-based Static Malware Classifiers**

cs.CR

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2303.13372v2) [paper-pdf](http://arxiv.org/pdf/2303.13372v2)

**Authors**: Shoumik Saha, Wenxiao Wang, Yigitcan Kaya, Soheil Feizi, Tudor Dumitras

**Abstract**: Malware detection has long been a stage for an ongoing arms race between malware authors and anti-virus systems. Solutions that utilize machine learning (ML) gain traction as the scale of this arms race increases. This trend, however, makes performing attacks directly on ML an attractive prospect for adversaries. We study this arms race from both perspectives in the context of MalConv, a popular convolutional neural network-based malware classifier that operates on raw bytes of files. First, we show that MalConv is vulnerable to adversarial patch attacks: appending a byte-level patch to malware files bypasses detection 94.3% of the time. Moreover, we develop a universal adversarial patch (UAP) attack where a single patch can drop the detection rate in constant time of any malware file that contains it by 80%. These patches are effective even being relatively small with respect to the original file size -- between 2%-8%. As a countermeasure, we then perform window ablation that allows us to apply de-randomized smoothing, a modern certified defense to patch attacks in vision tasks, to raw files. The resulting `smoothed-MalConv' can detect over 80% of malware that contains the universal patch and provides certified robustness up to 66%, outlining a promising step towards robust malware detection. To our knowledge, we are the first to apply universal adversarial patch attack and certified defense using ablations on byte level in the malware field.



## **24. CosPGD: a unified white-box adversarial attack for pixel-wise prediction tasks**

cs.CV

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2302.02213v2) [paper-pdf](http://arxiv.org/pdf/2302.02213v2)

**Authors**: Shashank Agnihotri, Steffen Jung, Margret Keuper

**Abstract**: While neural networks allow highly accurate predictions in many tasks, their lack of robustness towards even slight input perturbations hampers their deployment in many real-world applications. Recent research towards evaluating the robustness of neural networks such as the seminal projected gradient descent(PGD) attack and subsequent works have drawn significant attention, as they provide an effective insight into the quality of representations learned by the network. However, these methods predominantly focus on image classification tasks, while only a few approaches specifically address the analysis of pixel-wise prediction tasks such as semantic segmentation, optical flow, disparity estimation, and others, respectively. Thus, there is a lack of a unified adversarial robustness benchmarking tool(algorithm) that is applicable to all such pixel-wise prediction tasks. In this work, we close this gap and propose CosPGD, a novel white-box adversarial attack that allows optimizing dedicated attacks for any pixel-wise prediction task in a unified setting. It leverages the cosine similarity between the distributions over the predictions and ground truth (or target) to extend directly from classification tasks to regression settings. We outperform the SotA on semantic segmentation attacks in our experiments on PASCAL VOC2012 and CityScapes. Further, we set a new benchmark for adversarial attacks on optical flow, and image restoration displaying the ability to extend to any pixel-wise prediction task.



## **25. Replay-based Recovery for Autonomous Robotic Vehicles from Sensor Deception Attacks**

cs.RO

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2209.04554v4) [paper-pdf](http://arxiv.org/pdf/2209.04554v4)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstract**: Sensors are crucial for autonomous operation in robotic vehicles (RV). Unfortunately, RV sensors can be compromised by physical attacks such as tampering or spoofing, leading to a crash. In this paper, we present DeLorean, a modelfree recovery framework for recovering autonomous RVs from sensor deception attacks (SDA). DeLorean is designed to recover RVs even from a strong SDA in which the adversary targets multiple heterogeneous sensors simultaneously (even all the sensors). Under SDAs, DeLorean inspects the attack induced errors, identifies the targeted sensors, and prevents the erroneous sensor inputs from being used to derive actuator signals. DeLorean then replays historic state information in the RV's feedback control loop for a temporary mitigation and recovers the RV from SDA. Our evaluation on four real and two simulated RVs shows that DeLorean can recover RVs from SDAs, and ensure mission success in 90.7% of the cases on average.



## **26. Adversarial Training Should Be Cast as a Non-Zero-Sum Game**

cs.LG

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2306.11035v1) [paper-pdf](http://arxiv.org/pdf/2306.11035v1)

**Authors**: Alexander Robey, Fabian Latorre, George J. Pappas, Hamed Hassani, Volkan Cevher

**Abstract**: One prominent approach toward resolving the adversarial vulnerability of deep neural networks is the two-player zero-sum paradigm of adversarial training, in which predictors are trained against adversarially-chosen perturbations of data. Despite the promise of this approach, algorithms based on this paradigm have not engendered sufficient levels of robustness, and suffer from pathological behavior like robust overfitting. To understand this shortcoming, we first show that the commonly used surrogate-based relaxation used in adversarial training algorithms voids all guarantees on the robustness of trained classifiers. The identification of this pitfall informs a novel non-zero-sum bilevel formulation of adversarial training, wherein each player optimizes a different objective function. Our formulation naturally yields a simple algorithmic framework that matches and in some cases outperforms state-of-the-art attacks, attains comparable levels of robustness to standard adversarial training algorithms, and does not suffer from robust overfitting.



## **27. Eigenpatches -- Adversarial Patches from Principal Components**

cs.CV

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2306.10963v1) [paper-pdf](http://arxiv.org/pdf/2306.10963v1)

**Authors**: Jens Bayer, Stefan Becker, David Münch, Michael Arens

**Abstract**: Adversarial patches are still a simple yet powerful white box attack that can be used to fool object detectors by suppressing possible detections. The patches of these so-called evasion attacks are computational expensive to produce and require full access to the attacked detector. This paper addresses the problem of computational expensiveness by analyzing 375 generated patches, calculating the principal components of these and show, that linear combinations of the resulting "eigenpatches" can be used to fool object detections successfully.



## **28. Attack-Resilient Design for Connected and Automated Vehicles**

eess.SY

arXiv admin note: text overlap with arXiv:2109.01553

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2306.10925v1) [paper-pdf](http://arxiv.org/pdf/2306.10925v1)

**Authors**: Tianci Yang, Carlos Murguia, Dragan Nesic, Chau Yuen

**Abstract**: By sharing local sensor information via Vehicle-to-Vehicle (V2V) wireless communication networks, Cooperative Adaptive Cruise Control (CACC) is a technology that enables Connected and Automated Vehicles (CAVs) to drive autonomously on the highway in closely-coupled platoons. The use of CACC technologies increases safety and the traffic throughput, and decreases fuel consumption and CO2 emissions. However, CAVs heavily rely on embedded software, hardware, and communication networks that make them vulnerable to a range of cyberattacks. Cyberattacks to a particular CAV compromise the entire platoon as CACC schemes propagate corrupted data to neighboring vehicles potentially leading to traffic delays and collisions. Physics-based monitors can be used to detect the presence of False Data Injection (FDI) attacks to CAV sensors; however, unavoidable system disturbances and modelling uncertainty often translates to conservative detection results. Given enough system knowledge, adversaries are still able to launch a range of attacks that can surpass the detection scheme by hiding within the system disturbances and uncertainty -- we refer to this class of attacks as \textit{stealthy FDI attacks}. Stealthy attacks are hard to deal with as they affect the platoon dynamics without being noticed. In this manuscript, we propose a co-design methodology (built around a series convex programs) to synthesize distributed attack monitors and $H_{\infty}$ CACC controllers that minimize the joint effect of stealthy FDI attacks and system disturbances on the platoon dynamics while guaranteeing a prescribed platooning performance (in terms of tracking and string stability). Computer simulations are provided to illustrate the performance of out tools.



## **29. On the Robustness of Dataset Inference**

cs.LG

19 pages; Accepted to Transactions on Machine Learning Research  06/2023

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2210.13631v3) [paper-pdf](http://arxiv.org/pdf/2210.13631v3)

**Authors**: Sebastian Szyller, Rui Zhang, Jian Liu, N. Asokan

**Abstract**: Machine learning (ML) models are costly to train as they can require a significant amount of data, computational resources and technical expertise. Thus, they constitute valuable intellectual property that needs protection from adversaries wanting to steal them. Ownership verification techniques allow the victims of model stealing attacks to demonstrate that a suspect model was in fact stolen from theirs.   Although a number of ownership verification techniques based on watermarking or fingerprinting have been proposed, most of them fall short either in terms of security guarantees (well-equipped adversaries can evade verification) or computational cost. A fingerprinting technique, Dataset Inference (DI), has been shown to offer better robustness and efficiency than prior methods.   The authors of DI provided a correctness proof for linear (suspect) models. However, in a subspace of the same setting, we prove that DI suffers from high false positives (FPs) -- it can incorrectly identify an independent model trained with non-overlapping data from the same distribution as stolen. We further prove that DI also triggers FPs in realistic, non-linear suspect models. We then confirm empirically that DI in the black-box setting leads to FPs, with high confidence.   Second, we show that DI also suffers from false negatives (FNs) -- an adversary can fool DI (at the cost of incurring some accuracy loss) by regularising a stolen model's decision boundaries using adversarial training, thereby leading to an FN. To this end, we demonstrate that black-box DI fails to identify a model adversarially trained from a stolen dataset -- the setting where DI is the hardest to evade.   Finally, we discuss the implications of our findings, the viability of fingerprinting-based ownership verification in general, and suggest directions for future work.



## **30. Hidden Backdoor Attack against Deep Learning-Based Wireless Signal Modulation Classifiers**

eess.SP

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2306.10753v1) [paper-pdf](http://arxiv.org/pdf/2306.10753v1)

**Authors**: Yunsong Huang, Weicheng Liu, Hui-Ming Wang

**Abstract**: Recently, DL has been exploited in wireless communications such as modulation classification. However, due to the openness of wireless channel and unexplainability of DL, it is also vulnerable to adversarial attacks. In this correspondence, we investigate a so called hidden backdoor attack to modulation classification, where the adversary puts elaborately designed poisoned samples on the basis of IQ sequences into training dataset. These poisoned samples are hidden because it could not be found by traditional classification methods. And poisoned samples are same to samples with triggers which are patched samples in feature space. We show that the hidden backdoor attack can reduce the accuracy of modulation classification significantly with patched samples. At last, we propose activation cluster to detect abnormal samples in training dataset.



## **31. Adversarial Camouflage for Node Injection Attack on Graphs**

cs.LG

Submitted to Information Sciences. Code:  https://github.com/TaoShuchang/CANA

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2208.01819v3) [paper-pdf](http://arxiv.org/pdf/2208.01819v3)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Liang Hou, Fei Sun, Xueqi Cheng

**Abstract**: Node injection attacks on Graph Neural Networks (GNNs) have received emerging attention due to their potential to significantly degrade GNN performance with high attack success rates. However, our study indicates these attacks often fail in practical scenarios, since defense/detection methods can easily identify and remove the injected nodes. To address this, we devote to camouflage node injection attack, making injected nodes appear normal and imperceptible to defense/detection methods. Unfortunately, the non-Euclidean nature of graph data and lack of intuitive prior present great challenges to the formalization, implementation, and evaluation of camouflage. In this paper, we first propose and define camouflage as distribution similarity between ego networks of injected nodes and normal nodes. Then for implementation, we propose an adversarial CAmouflage framework for Node injection Attack, namely CANA, to improve attack performance under defense/detection methods in practical scenarios. A novel camouflage metric is further designed under the guide of distribution similarity. Extensive experiments demonstrate that CANA can significantly improve the attack performance under defense/detection methods with higher camouflage or imperceptibility. This work urges us to raise awareness of the security vulnerabilities of GNNs in practical applications. The implementation of CANA is available at https://github.com/TaoShuchang/CANA.



## **32. Intriguing Properties of Text-guided Diffusion Models**

cs.CV

Project page: https://sage-diffusion.github.io/

**SubmitDate**: 2023-06-18    [abs](http://arxiv.org/abs/2306.00974v3) [paper-pdf](http://arxiv.org/pdf/2306.00974v3)

**Authors**: Qihao Liu, Adam Kortylewski, Yutong Bai, Song Bai, Alan Yuille

**Abstract**: Text-guided diffusion models (TDMs) are widely applied but can fail unexpectedly. Common failures include: (i) natural-looking text prompts generating images with the wrong content, or (ii) different random samples of the latent variables that generate vastly different, and even unrelated, outputs despite being conditioned on the same text prompt. In this work, we aim to study and understand the failure modes of TDMs in more detail. To achieve this, we propose SAGE, an adversarial attack on TDMs that uses image classifiers as surrogate loss functions, to search over the discrete prompt space and the high-dimensional latent space of TDMs to automatically discover unexpected behaviors and failure cases in the image generation. We make several technical contributions to ensure that SAGE finds failure cases of the diffusion model, rather than the classifier, and verify this in a human study. Our study reveals four intriguing properties of TDMs that have not been systematically studied before: (1) We find a variety of natural text prompts producing images that fail to capture the semantics of input texts. We categorize these failures into ten distinct types based on the underlying causes. (2) We find samples in the latent space (which are not outliers) that lead to distorted images independent of the text prompt, suggesting that parts of the latent space are not well-structured. (3) We also find latent samples that lead to natural-looking images which are unrelated to the text prompt, implying a potential misalignment between the latent and prompt spaces. (4) By appending a single adversarial token embedding to an input prompt we can generate a variety of specified target objects, while only minimally affecting the CLIP score. This demonstrates the fragility of language representations and raises potential safety concerns.



## **33. Towards A Proactive ML Approach for Detecting Backdoor Poison Samples**

cs.LG

USENIX Security 2023

**SubmitDate**: 2023-06-18    [abs](http://arxiv.org/abs/2205.13616v3) [paper-pdf](http://arxiv.org/pdf/2205.13616v3)

**Authors**: Xiangyu Qi, Tinghao Xie, Jiachen T. Wang, Tong Wu, Saeed Mahloujifar, Prateek Mittal

**Abstract**: Adversaries can embed backdoors in deep learning models by introducing backdoor poison samples into training datasets. In this work, we investigate how to detect such poison samples to mitigate the threat of backdoor attacks. First, we uncover a post-hoc workflow underlying most prior work, where defenders passively allow the attack to proceed and then leverage the characteristics of the post-attacked model to uncover poison samples. We reveal that this workflow does not fully exploit defenders' capabilities, and defense pipelines built on it are prone to failure or performance degradation in many scenarios. Second, we suggest a paradigm shift by promoting a proactive mindset in which defenders engage proactively with the entire model training and poison detection pipeline, directly enforcing and magnifying distinctive characteristics of the post-attacked model to facilitate poison detection. Based on this, we formulate a unified framework and provide practical insights on designing detection pipelines that are more robust and generalizable. Third, we introduce the technique of Confusion Training (CT) as a concrete instantiation of our framework. CT applies an additional poisoning attack to the already poisoned dataset, actively decoupling benign correlation while exposing backdoor patterns to detection. Empirical evaluations on 4 datasets and 14 types of attacks validate the superiority of CT over 14 baseline defenses.



## **34. Adversaries with Limited Information in the Friedkin--Johnsen Model**

cs.SI

To appear at KDD'23

**SubmitDate**: 2023-06-17    [abs](http://arxiv.org/abs/2306.10313v1) [paper-pdf](http://arxiv.org/pdf/2306.10313v1)

**Authors**: Sijing Tu, Stefan Neumann, Aristides Gionis

**Abstract**: In recent years, online social networks have been the target of adversaries who seek to introduce discord into societies, to undermine democracies and to destabilize communities. Often the goal is not to favor a certain side of a conflict but to increase disagreement and polarization. To get a mathematical understanding of such attacks, researchers use opinion-formation models from sociology, such as the Friedkin--Johnsen model, and formally study how much discord the adversary can produce when altering the opinions for only a small set of users. In this line of work, it is commonly assumed that the adversary has full knowledge about the network topology and the opinions of all users. However, the latter assumption is often unrealistic in practice, where user opinions are not available or simply difficult to estimate accurately.   To address this concern, we raise the following question: Can an attacker sow discord in a social network, even when only the network topology is known? We answer this question affirmatively. We present approximation algorithms for detecting a small set of users who are highly influential for the disagreement and polarization in the network. We show that when the adversary radicalizes these users and if the initial disagreement/polarization in the network is not very high, then our method gives a constant-factor approximation on the setting when the user opinions are known. To find the set of influential users, we provide a novel approximation algorithm for a variant of MaxCut in graphs with positive and negative edge weights. We experimentally evaluate our methods, which have access only to the network topology, and we find that they have similar performance as methods that have access to the network topology and all user opinions. We further present an NP-hardness proof, which was an open question by Chen and Racz [IEEE Trans. Netw. Sci. Eng., 2021].



## **35. Edge Learning for 6G-enabled Internet of Things: A Comprehensive Survey of Vulnerabilities, Datasets, and Defenses**

cs.CR

**SubmitDate**: 2023-06-17    [abs](http://arxiv.org/abs/2306.10309v1) [paper-pdf](http://arxiv.org/pdf/2306.10309v1)

**Authors**: Mohamed Amine Ferrag, Othmane Friha, Burak Kantarci, Norbert Tihanyi, Lucas Cordeiro, Merouane Debbah, Djallel Hamouda, Muna Al-Hawawreh, Kim-Kwang Raymond Choo

**Abstract**: The ongoing deployment of the fifth generation (5G) wireless networks constantly reveals limitations concerning its original concept as a key driver of Internet of Everything (IoE) applications. These 5G challenges are behind worldwide efforts to enable future networks, such as sixth generation (6G) networks, to efficiently support sophisticated applications ranging from autonomous driving capabilities to the Metaverse. Edge learning is a new and powerful approach to training models across distributed clients while protecting the privacy of their data. This approach is expected to be embedded within future network infrastructures, including 6G, to solve challenging problems such as resource management and behavior prediction. This survey article provides a holistic review of the most recent research focused on edge learning vulnerabilities and defenses for 6G-enabled IoT. We summarize the existing surveys on machine learning for 6G IoT security and machine learning-associated threats in three different learning modes: centralized, federated, and distributed. Then, we provide an overview of enabling emerging technologies for 6G IoT intelligence. Moreover, we provide a holistic survey of existing research on attacks against machine learning and classify threat models into eight categories, including backdoor attacks, adversarial examples, combined attacks, poisoning attacks, Sybil attacks, byzantine attacks, inference attacks, and dropping attacks. In addition, we provide a comprehensive and detailed taxonomy and a side-by-side comparison of the state-of-the-art defense methods against edge learning vulnerabilities. Finally, as new attacks and defense technologies are realized, new research and future overall prospects for 6G-enabled IoT are discussed.



## **36. You Don't Need Robust Machine Learning to Manage Adversarial Attack Risks**

cs.LG

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2306.09951v1) [paper-pdf](http://arxiv.org/pdf/2306.09951v1)

**Authors**: Edward Raff, Michel Benaroch, Andrew L. Farris

**Abstract**: The robustness of modern machine learning (ML) models has become an increasing concern within the community. The ability to subvert a model into making errant predictions using seemingly inconsequential changes to input is startling, as is our lack of success in building models robust to this concern. Existing research shows progress, but current mitigations come with a high cost and simultaneously reduce the model's accuracy. However, such trade-offs may not be necessary when other design choices could subvert the risk. In this survey we review the current literature on attacks and their real-world occurrences, or limited evidence thereof, to critically evaluate the real-world risks of adversarial machine learning (AML) for the average entity. This is done with an eye toward how one would then mitigate these attacks in practice, the risks for production deployment, and how those risks could be managed. In doing so we elucidate that many AML threats do not warrant the cost and trade-offs of robustness due to a low likelihood of attack or availability of superior non-ML mitigations. Our analysis also recommends cases where an actor should be concerned about AML to the degree where robust ML models are necessary for a complete deployment.



## **37. Adversarial Cheap Talk**

cs.LG

To be published at ICML 2023. Project video and code are available at  https://sites.google.com/view/adversarial-cheap-talk

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2211.11030v3) [paper-pdf](http://arxiv.org/pdf/2211.11030v3)

**Authors**: Chris Lu, Timon Willi, Alistair Letcher, Jakob Foerster

**Abstract**: Adversarial attacks in reinforcement learning (RL) often assume highly-privileged access to the victim's parameters, environment, or data. Instead, this paper proposes a novel adversarial setting called a Cheap Talk MDP in which an Adversary can merely append deterministic messages to the Victim's observation, resulting in a minimal range of influence. The Adversary cannot occlude ground truth, influence underlying environment dynamics or reward signals, introduce non-stationarity, add stochasticity, see the Victim's actions, or access their parameters. Additionally, we present a simple meta-learning algorithm called Adversarial Cheap Talk (ACT) to train Adversaries in this setting. We demonstrate that an Adversary trained with ACT still significantly influences the Victim's training and testing performance, despite the highly constrained setting. Affecting train-time performance reveals a new attack vector and provides insight into the success and failure modes of existing RL algorithms. More specifically, we show that an ACT Adversary is capable of harming performance by interfering with the learner's function approximation, or instead helping the Victim's performance by outputting useful features. Finally, we show that an ACT Adversary can manipulate messages during train-time to directly and arbitrarily control the Victim at test-time. Project video and code are available at https://sites.google.com/view/adversarial-cheap-talk



## **38. Query-Free Evasion Attacks Against Machine Learning-Based Malware Detectors with Generative Adversarial Networks**

cs.CR

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2306.09925v1) [paper-pdf](http://arxiv.org/pdf/2306.09925v1)

**Authors**: Daniel Gibert, Jordi Planes, Quan Le, Giulio Zizzo

**Abstract**: Malware detectors based on machine learning (ML) have been shown to be susceptible to adversarial malware examples. However, current methods to generate adversarial malware examples still have their limits. They either rely on detailed model information (gradient-based attacks), or on detailed outputs of the model - such as class probabilities (score-based attacks), neither of which are available in real-world scenarios. Alternatively, adversarial examples might be crafted using only the label assigned by the detector (label-based attack) to train a substitute network or an agent using reinforcement learning. Nonetheless, label-based attacks might require querying a black-box system from a small number to thousands of times, depending on the approach, which might not be feasible against malware detectors. This work presents a novel query-free approach to craft adversarial malware examples to evade ML-based malware detectors. To this end, we have devised a GAN-based framework to generate adversarial malware examples that look similar to benign executables in the feature space. To demonstrate the suitability of our approach we have applied the GAN-based attack to three common types of features usually employed by static ML-based malware detectors: (1) Byte histogram features, (2) API-based features, and (3) String-based features. Results show that our model-agnostic approach performs on par with MalGAN, while generating more realistic adversarial malware examples without requiring any query to the malware detectors. Furthermore, we have tested the generated adversarial examples against state-of-the-art multimodal and deep learning malware detectors, showing a decrease in detection performance, as well as a decrease in the average number of detections by the anti-malware engines in VirusTotal.



## **39. Wasserstein distributional robustness of neural networks**

cs.LG

23 pages, 6 figures, 8 tables

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2306.09844v1) [paper-pdf](http://arxiv.org/pdf/2306.09844v1)

**Authors**: Xingjian Bai, Guangyi He, Yifan Jiang, Jan Obloj

**Abstract**: Deep neural networks are known to be vulnerable to adversarial attacks (AA). For an image recognition task, this means that a small perturbation of the original can result in the image being misclassified. Design of such attacks as well as methods of adversarial training against them are subject of intense research. We re-cast the problem using techniques of Wasserstein distributionally robust optimization (DRO) and obtain novel contributions leveraging recent insights from DRO sensitivity analysis. We consider a set of distributional threat models. Unlike the traditional pointwise attacks, which assume a uniform bound on perturbation of each input data point, distributional threat models allow attackers to perturb inputs in a non-uniform way. We link these more general attacks with questions of out-of-sample performance and Knightian uncertainty. To evaluate the distributional robustness of neural networks, we propose a first-order AA algorithm and its multi-step version. Our attack algorithms include Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) as special cases. Furthermore, we provide a new asymptotic estimate of the adversarial accuracy against distributional threat models. The bound is fast to compute and first-order accurate, offering new insights even for the pointwise AA. It also naturally yields out-of-sample performance guarantees. We conduct numerical experiments on the CIFAR-10 dataset using DNNs on RobustBench to illustrate our theoretical results. Our code is available at https://github.com/JanObloj/W-DRO-Adversarial-Methods.



## **40. TransFool: An Adversarial Attack against Neural Machine Translation Models**

cs.CL

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2302.00944v2) [paper-pdf](http://arxiv.org/pdf/2302.00944v2)

**Authors**: Sahar Sadrizadeh, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Deep neural networks have been shown to be vulnerable to small perturbations of their inputs, known as adversarial attacks. In this paper, we investigate the vulnerability of Neural Machine Translation (NMT) models to adversarial attacks and propose a new attack algorithm called TransFool. To fool NMT models, TransFool builds on a multi-term optimization problem and a gradient projection step. By integrating the embedding representation of a language model, we generate fluent adversarial examples in the source language that maintain a high level of semantic similarity with the clean samples. Experimental results demonstrate that, for different translation tasks and NMT architectures, our white-box attack can severely degrade the translation quality while the semantic similarity between the original and the adversarial sentences stays high. Moreover, we show that TransFool is transferable to unknown target models. Finally, based on automatic and human evaluations, TransFool leads to improvement in terms of success rate, semantic similarity, and fluency compared to the existing attacks both in white-box and black-box settings. Thus, TransFool permits us to better characterize the vulnerability of NMT models and outlines the necessity to design strong defense mechanisms and more robust NMT systems for real-life applications.



## **41. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

cs.CR

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2209.03755v4) [paper-pdf](http://arxiv.org/pdf/2209.03755v4)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstract**: Mis- and disinformation are a substantial global threat to our security and safety. To cope with the scale of online misinformation, researchers have been working on automating fact-checking by retrieving and verifying against relevant evidence. However, despite many advances, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence and generate diverse and claim-aligned evidence. Thus, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.



## **42. Adversarial Image Color Transformations in Explicit Color Filter Space**

cs.CV

Published at IEEE Transactions on Information Forensics and Security  2023. Code is available at  https://github.com/ZhengyuZhao/ACE/tree/master/Journal_version

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2011.06690v3) [paper-pdf](http://arxiv.org/pdf/2011.06690v3)

**Authors**: Zhengyu Zhao, Zhuoran Liu, Martha Larson

**Abstract**: Deep Neural Networks have been shown to be vulnerable to adversarial images. Conventional attacks strive for indistinguishable adversarial images with strictly restricted perturbations. Recently, researchers have moved to explore distinguishable yet non-suspicious adversarial images and demonstrated that color transformation attacks are effective. In this work, we propose Adversarial Color Filter (AdvCF), a novel color transformation attack that is optimized with gradient information in the parameter space of a simple color filter. In particular, our color filter space is explicitly specified so that we are able to provide a systematic analysis of model robustness against adversarial color transformations, from both the attack and defense perspectives. In contrast, existing color transformation attacks do not offer the opportunity for systematic analysis due to the lack of such an explicit space. We further demonstrate the effectiveness of our AdvCF in fooling image classifiers and also compare it with other color transformation attacks regarding their robustness to defenses and image acceptability through an extensive user study. We also highlight the human-interpretability of AdvCF and show its superiority over the state-of-the-art human-interpretable color transformation attack on both image acceptability and efficiency. Additional results provide interesting new insights into model robustness against AdvCF in another three visual tasks.



## **43. Distributed Energy Resources Cybersecurity Outlook: Vulnerabilities, Attacks, Impacts, and Mitigations**

cs.CR

IEEE Systems Journal

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2205.11171v3) [paper-pdf](http://arxiv.org/pdf/2205.11171v3)

**Authors**: Ioannis Zografopoulos, Nikos D. Hatziargyriou, Charalambos Konstantinou

**Abstract**: The digitization and decentralization of the electric power grid are key thrusts for an economically and environmentally sustainable future. Towards this goal, distributed energy resources (DER), including rooftop solar panels, battery storage, electric vehicles, etc., are becoming ubiquitous in power systems. Power utilities benefit from DERs as they minimize operational costs; at the same time, DERs grant users and aggregators control over the power they produce and consume. DERs are interconnected, interoperable, and support remotely controllable features, thus, their cybersecurity is of cardinal importance. DER communication dependencies and the diversity of DER architectures widen the threat surface and aggravate the cybersecurity posture of power systems. In this work, we focus on security oversights that reside in the cyber and physical layers of DERs and can jeopardize grid operations. Existing works have underlined the impact of cyberattacks targeting DER assets, however, they either focus on specific system components (e.g., communication protocols), do not consider the mission-critical objectives of DERs, or neglect the adversarial perspective (e.g., adversary/attack models) altogether. To address these omissions, we comprehensively analyze adversarial capabilities and objectives when manipulating DER assets, and then present how protocol and device-level vulnerabilities can materialize into cyberattacks impacting power system operations. Finally, we provide mitigation strategies to thwart adversaries and directions for future DER cybersecurity research.



## **44. Inroads into Autonomous Network Defence using Explained Reinforcement Learning**

cs.CR

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.09318v1) [paper-pdf](http://arxiv.org/pdf/2306.09318v1)

**Authors**: Myles Foley, Mia Wang, Zoe M, Chris Hicks, Vasilios Mavroudis

**Abstract**: Computer network defence is a complicated task that has necessitated a high degree of human involvement. However, with recent advancements in machine learning, fully autonomous network defence is becoming increasingly plausible. This paper introduces an end-to-end methodology for studying attack strategies, designing defence agents and explaining their operation. First, using state diagrams, we visualise adversarial behaviour to gain insight about potential points of intervention and inform the design of our defensive models. We opt to use a set of deep reinforcement learning agents trained on different parts of the task and organised in a shallow hierarchy. Our evaluation shows that the resulting design achieves a substantial performance improvement compared to prior work. Finally, to better investigate the decision-making process of our agents, we complete our analysis with a feature ablation and importance study.



## **45. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks in the Physical World**

cs.CV

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.09124v1) [paper-pdf](http://arxiv.org/pdf/2306.09124v1)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks in the physical world, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications, yet current research in this area is severely lacking. In this paper, we propose DIFFender, a novel defense method that leverages the pre-trained diffusion model to perform both localization and defense against potential adversarial patch attacks. DIFFender is designed as a pipeline consisting of two main stages: patch localization and restoration. In the localization stage, we exploit the intriguing properties of a diffusion model to effectively identify the locations of adversarial patches. In the restoration stage, we employ a text-guided diffusion model to eliminate adversarial regions in the image while preserving the integrity of the visual content. Additionally, we design a few-shot prompt-tuning algorithm to facilitate simple and efficient tuning, enabling the learned representations to easily transfer to downstream tasks, which optimize two stages jointly. We conduct extensive experiments on image classification and face recognition to demonstrate that DIFFender exhibits superior robustness under strong adaptive attacks and generalizes well across various scenarios, diverse classifiers, and multiple attack methods.



## **46. The Effect of Length on Key Fingerprint Verification Security and Usability**

cs.CR

Accepted to International Conference on Availability, Reliability and  Security (ARES 2023)

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.04574v2) [paper-pdf](http://arxiv.org/pdf/2306.04574v2)

**Authors**: Dan Turner, Siamak F. Shahandashti, Helen Petrie

**Abstract**: In applications such as end-to-end encrypted instant messaging, secure email, and device pairing, users need to compare key fingerprints to detect impersonation and adversary-in-the-middle attacks. Key fingerprints are usually computed as truncated hashes of each party's view of the channel keys, encoded as an alphanumeric or numeric string, and compared out-of-band, e.g. manually, to detect any inconsistencies. Previous work has extensively studied the usability of various verification strategies and encoding formats, however, the exact effect of key fingerprint length on the security and usability of key fingerprint verification has not been rigorously investigated. We present a 162-participant study on the effect of numeric key fingerprint length on comparison time and error rate. While the results confirm some widely-held intuitions such as general comparison times and errors increasing significantly with length, a closer look reveals interesting nuances. The significant rise in comparison time only occurs when highly similar fingerprints are compared, and comparison time remains relatively constant otherwise. On errors, our results clearly distinguish between security non-critical errors that remain low irrespective of length and security critical errors that significantly rise, especially at higher fingerprint lengths. A noteworthy implication of this latter result is that Signal/WhatsApp key fingerprints provide a considerably lower level of security than usually assumed.



## **47. Community Detection Attack against Collaborative Learning-based Recommender Systems**

cs.IR

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.08929v1) [paper-pdf](http://arxiv.org/pdf/2306.08929v1)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning based recommender systems emerged following the success of collaborative learning techniques such as Federated Learning (FL) and Gossip Learning (GL). In these systems, users participate in the training of a recommender system while keeping their history of consumed items on their devices. While these solutions seemed appealing for preserving the privacy of the participants at a first glance, recent studies have shown that collaborative learning can be vulnerable to a variety of privacy attacks. In this paper we propose a novel privacy attack called Community Detection Attack (CDA), which allows an adversary to discover the members of a community based on a set of items of her choice (e.g., discovering users interested in LGBT content). Through experiments on three real recommendation datasets and by using two state-of-the-art recommendation models, we assess the sensitivity of an FL-based recommender system as well as two flavors of Gossip Learning-based recommender systems to CDA. Results show that on all models and all datasets, the FL setting is more vulnerable to CDA than Gossip settings. We further evaluated two off-the-shelf mitigation strategies, namely differential privacy (DP) and a share less policy, which consists in sharing a subset of model parameters. Results show a better privacy-utility trade-off for the share less policy compared to DP especially in the Gossip setting.



## **48. MalProtect: Stateful Defense Against Adversarial Query Attacks in ML-based Malware Detection**

cs.LG

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2302.10739v2) [paper-pdf](http://arxiv.org/pdf/2302.10739v2)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: ML models are known to be vulnerable to adversarial query attacks. In these attacks, queries are iteratively perturbed towards a particular class without any knowledge of the target model besides its output. The prevalence of remotely-hosted ML classification models and Machine-Learning-as-a-Service platforms means that query attacks pose a real threat to the security of these systems. To deal with this, stateful defenses have been proposed to detect query attacks and prevent the generation of adversarial examples by monitoring and analyzing the sequence of queries received by the system. Several stateful defenses have been proposed in recent years. However, these defenses rely solely on similarity or out-of-distribution detection methods that may be effective in other domains. In the malware detection domain, the methods to generate adversarial examples are inherently different, and therefore we find that such detection mechanisms are significantly less effective. Hence, in this paper, we present MalProtect, which is a stateful defense against query attacks in the malware detection domain. MalProtect uses several threat indicators to detect attacks. Our results show that it reduces the evasion rate of adversarial query attacks by 80+\% in Android and Windows malware, across a range of attacker scenarios. In the first evaluation of its kind, we show that MalProtect outperforms prior stateful defenses, especially under the peak adversarial threat.



## **49. Augment then Smooth: Reconciling Differential Privacy with Certified Robustness**

cs.LG

25 pages, 19 figures

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08656v1) [paper-pdf](http://arxiv.org/pdf/2306.08656v1)

**Authors**: Jiapeng Wu, Atiyeh Ashari Ghomi, David Glukhov, Jesse C. Cresswell, Franziska Boenisch, Nicolas Papernot

**Abstract**: Machine learning models are susceptible to a variety of attacks that can erode trust in their deployment. These threats include attacks against the privacy of training data and adversarial examples that jeopardize model accuracy. Differential privacy and randomized smoothing are effective defenses that provide certifiable guarantees for each of these threats, however, it is not well understood how implementing either defense impacts the other. In this work, we argue that it is possible to achieve both privacy guarantees and certified robustness simultaneously. We provide a framework called DP-CERT for integrating certified robustness through randomized smoothing into differentially private model training. For instance, compared to differentially private stochastic gradient descent on CIFAR10, DP-CERT leads to a 12-fold increase in certified accuracy and a 10-fold increase in the average certified radius at the expense of a drop in accuracy of 1.2%. Through in-depth per-sample metric analysis, we show that the certified radius correlates with the local Lipschitz constant and smoothness of the loss surface. This provides a new way to diagnose when private models will fail to be robust.



## **50. A Unified Framework of Graph Information Bottleneck for Robustness and Membership Privacy**

cs.LG

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08604v1) [paper-pdf](http://arxiv.org/pdf/2306.08604v1)

**Authors**: Enyan Dai, Limeng Cui, Zhengyang Wang, Xianfeng Tang, Yinghan Wang, Monica Cheng, Bing Yin, Suhang Wang

**Abstract**: Graph Neural Networks (GNNs) have achieved great success in modeling graph-structured data. However, recent works show that GNNs are vulnerable to adversarial attacks which can fool the GNN model to make desired predictions of the attacker. In addition, training data of GNNs can be leaked under membership inference attacks. This largely hinders the adoption of GNNs in high-stake domains such as e-commerce, finance and bioinformatics. Though investigations have been made in conducting robust predictions and protecting membership privacy, they generally fail to simultaneously consider the robustness and membership privacy. Therefore, in this work, we study a novel problem of developing robust and membership privacy-preserving GNNs. Our analysis shows that Information Bottleneck (IB) can help filter out noisy information and regularize the predictions on labeled samples, which can benefit robustness and membership privacy. However, structural noises and lack of labels in node classification challenge the deployment of IB on graph-structured data. To mitigate these issues, we propose a novel graph information bottleneck framework that can alleviate structural noises with neighbor bottleneck. Pseudo labels are also incorporated in the optimization to minimize the gap between the predictions on the labeled set and unlabeled set for membership privacy. Extensive experiments on real-world datasets demonstrate that our method can give robust predictions and simultaneously preserve membership privacy.



