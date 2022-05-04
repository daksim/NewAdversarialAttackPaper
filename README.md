# Latest Adversarial Attack Papers
**update at 2022-05-05 06:31:51**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Adversarial Training for High-Stakes Reliability**

cs.LG

31 pages, 6 figures

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01663v1)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstracts**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a language generation task as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our simple "avoid injuries" task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. With our chosen thresholds, filtering with our baseline classifier decreases the rate of unsafe completions from about 2.4% to 0.003% on in-distribution data, which is near the limit of our ability to measure. We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on, without affecting in-distribution performance. We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.



## **2. A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space**

cs.AI

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2112.01156v2)

**Authors**: Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon

**Abstracts**: The generation of feasible adversarial examples is necessary for properly assessing models that work in constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective in four different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.



## **3. On the uncertainty principle of neural networks**

cs.LG

8 pages, 8 figures

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01493v1)

**Authors**: Jun-Jie Zhang, Dong-Xiao Zhang, Jian-Nan Chen, Long-Gang Pang

**Abstracts**: Despite the successes in many fields, it is found that neural networks are vulnerability and difficult to be both accurate and robust (robust means that the prediction of the trained network stays unchanged for inputs with non-random perturbations introduced by adversarial attacks). Various empirical and analytic studies have suggested that there is more or less a trade-off between the accuracy and robustness of neural networks. If the trade-off is inherent, applications based on the neural networks are vulnerable with untrustworthy predictions. It is then essential to ask whether the trade-off is an inherent property or not. Here, we show that the accuracy-robustness trade-off is an intrinsic property whose underlying mechanism is deeply related to the uncertainty principle in quantum mechanics. We find that for a neural network to be both accurate and robust, it needs to resolve the features of the two conjugated parts $x$ (the inputs) and $\Delta$ (the derivatives of the normalized loss function $J$ with respect to $x$), respectively. Analogous to the position-momentum conjugation in quantum mechanics, we show that the inputs and their conjugates cannot be resolved by a neural network simultaneously.



## **4. Self-Ensemble Adversarial Training for Improved Robustness**

cs.LG

18 pages, 3 figures, ICLR 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2203.09678v2)

**Authors**: Hongjun Wang, Yisen Wang

**Abstracts**: Due to numerous breakthroughs in real-world applications brought by machine intelligence, deep neural networks (DNNs) are widely employed in critical applications. However, predictions of DNNs are easily manipulated with imperceptible adversarial perturbations, which impedes the further deployment of DNNs and may result in profound security and privacy implications. By incorporating adversarial samples into the training data pool, adversarial training is the strongest principled strategy against various adversarial attacks among all sorts of defense methods. Recent works mainly focus on developing new loss functions or regularizers, attempting to find the unique optimal point in the weight space. But none of them taps the potentials of classifiers obtained from standard adversarial training, especially states on the searching trajectory of training. In this work, we are dedicated to the weight states of models through the training process and devise a simple but powerful \emph{Self-Ensemble Adversarial Training} (SEAT) method for yielding a robust classifier by averaging weights of history models. This considerably improves the robustness of the target model against several well known adversarial attacks, even merely utilizing the naive cross-entropy loss to supervise. We also discuss the relationship between the ensemble of predictions from different adversarially trained models and the prediction of weight-ensembled models, as well as provide theoretical and empirical evidence that the proposed self-ensemble method provides a smoother loss landscape and better robustness than both individual models and the ensemble of predictions from different classifiers. We further analyze a subtle but fatal issue in the general settings for the self-ensemble model, which causes the deterioration of the weight-ensembled method in the late phases.



## **5. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01287v1)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.



## **6. Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection**

cs.CV

CVPR 2022 camera ready

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2112.04532v2)

**Authors**: Jiang Liu, Alexander Levine, Chun Pong Lau, Rama Chellappa, Soheil Feizi

**Abstracts**: Object detection plays a key role in many security-critical systems. Adversarial patch attacks, which are easy to implement in the physical world, pose a serious threat to state-of-the-art object detectors. Developing reliable defenses for object detectors against patch attacks is critical but severely understudied. In this paper, we propose Segment and Complete defense (SAC), a general framework for defending object detectors against patch attacks through detection and removal of adversarial patches. We first train a patch segmenter that outputs patch masks which provide pixel-level localization of adversarial patches. We then propose a self adversarial training algorithm to robustify the patch segmenter. In addition, we design a robust shape completion algorithm, which is guaranteed to remove the entire patch from the images if the outputs of the patch segmenter are within a certain Hamming distance of the ground-truth patch masks. Our experiments on COCO and xView datasets demonstrate that SAC achieves superior robustness even under strong adaptive attacks with no reduction in performance on clean images, and generalizes well to unseen patch shapes, attack budgets, and unseen attack methods. Furthermore, we present the APRICOT-Mask dataset, which augments the APRICOT dataset with pixel-level annotations of adversarial patches. We show SAC can significantly reduce the targeted attack success rate of physical patch attacks. Our code is available at https://github.com/joellliu/SegmentAndComplete.



## **7. Defending Against Advanced Persistent Threats using Game-Theory**

cs.CR

preprint of a correction to the article with the same name, published  with PLOS ONE, and currently under review

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00956v1)

**Authors**: Stefan Rass, Sandra König, Stefan Schauer

**Abstracts**: Advanced persistent threats (APT) combine a variety of different attack forms ranging from social engineering to technical exploits. The diversity and usual stealthiness of APT turns them into a central problem of contemporary practical system security, since information on attacks, the current system status or the attacker's incentives is often vague, uncertain and in many cases even unavailable. Game theory is a natural approach to model the conflict between the attacker and the defender, and this work investigates a generalized class of matrix games as a risk mitigation tool for an APT defense. Unlike standard game and decision theory, our model is tailored to capture and handle the full uncertainty that is immanent to APT, such as disagreement among qualitative expert risk assessments, unknown adversarial incentives and uncertainty about the current system state (in terms of how deeply the attacker may have penetrated into the system's protective shells already). Practically, game-theoretic APT models can be derived straightforwardly from topological vulnerability analysis, together with risk assessments as they are done in common risk management standards like the ISO 31000 family. Theoretically, these models come with different properties than classical game theoretic models, whose technical solution presented in this work may be of independent interest.



## **8. BERTops: Studying BERT Representations under a Topological Lens**

cs.LG

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00953v1)

**Authors**: Jatin Chauhan, Manohar Kaul

**Abstracts**: Proposing scoring functions to effectively understand, analyze and learn various properties of high dimensional hidden representations of large-scale transformer models like BERT can be a challenging task. In this work, we explore a new direction by studying the topological features of BERT hidden representations using persistent homology (PH). We propose a novel scoring function named "persistence scoring function (PSF)" which: (i) accurately captures the homology of the high-dimensional hidden representations and correlates well with the test set accuracy of a wide range of datasets and outperforms existing scoring metrics, (ii) captures interesting post fine-tuning "per-class" level properties from both qualitative and quantitative viewpoints, (iii) is more stable to perturbations as compared to the baseline functions, which makes it a very robust proxy, and (iv) finally, also serves as a predictor of the attack success rates for a wide category of black-box and white-box adversarial attack methods. Our extensive correlation experiments demonstrate the practical utility of PSF on various NLP tasks relevant to BERT.



## **9. Revisiting Gaussian Neurons for Online Clustering with Unknown Number of Clusters**

cs.LG

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00920v1)

**Authors**: Ole Christian Eidheim

**Abstracts**: Despite the recent success of artificial neural networks, more biologically plausible learning methods may be needed to resolve the weaknesses of backpropagation trained models such as catastrophic forgetting and adversarial attacks. A novel local learning rule is presented that performs online clustering with a maximum limit of the number of cluster to be found rather than a fixed cluster count. Instead of using orthogonal weight or output activation constraints, activation sparsity is achieved by mutual repulsion of lateral Gaussian neurons ensuring that multiple neuron centers cannot occupy the same location in the input domain. An update method is also presented for adjusting the widths of the Gaussian neurons in cases where the data samples can be represented by means and variances. The algorithms were applied on the MNIST and CIFAR-10 datasets to create filters capturing the input patterns of pixel patches of various sizes. The experimental results demonstrate stability in the learned parameters across a large number of training samples.



## **10. Deep-Attack over the Deep Reinforcement Learning**

cs.LG

Accepted to Knowledge-Based Systems

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00807v1)

**Authors**: Yang Li, Quan Pan, Erik Cambria

**Abstracts**: Recent adversarial attack developments have made reinforcement learning more vulnerable, and different approaches exist to deploy attacks against it, where the key is how to choose the right timing of the attack. Some work tries to design an attack evaluation function to select critical points that will be attacked if the value is greater than a certain threshold. This approach makes it difficult to find the right place to deploy an attack without considering the long-term impact. In addition, there is a lack of appropriate indicators of assessment during attacks. To make the attacks more intelligent as well as to remedy the existing problems, we propose the reinforcement learning-based attacking framework by considering the effectiveness and stealthy spontaneously, while we also propose a new metric to evaluate the performance of the attack model in these two aspects. Experimental results show the effectiveness of our proposed model and the goodness of our proposed evaluation metric. Furthermore, we validate the transferability of the model, and also its robustness under the adversarial training.



## **11. Enhancing Adversarial Training with Feature Separability**

cs.CV

10 pages

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00637v1)

**Authors**: Yaxin Li, Xiaorui Liu, Han Xu, Wentao Wang, Jiliang Tang

**Abstracts**: Deep Neural Network (DNN) are vulnerable to adversarial attacks. As a countermeasure, adversarial training aims to achieve robustness based on the min-max optimization problem and it has shown to be one of the most effective defense strategies. However, in this work, we found that compared with natural training, adversarial training fails to learn better feature representations for either clean or adversarial samples, which can be one reason why adversarial training tends to have severe overfitting issues and less satisfied generalize performance. Specifically, we observe two major shortcomings of the features learned by existing adversarial training methods:(1) low intra-class feature similarity; and (2) conservative inter-classes feature variance. To overcome these shortcomings, we introduce a new concept of adversarial training graph (ATG) with which the proposed adversarial training with feature separability (ATFS) enables to coherently boost the intra-class feature similarity and increase inter-class feature variance. Through comprehensive experiments, we demonstrate that the proposed ATFS framework significantly improves both clean and robust performance.



## **12. Robust Fine-tuning via Perturbation and Interpolation from In-batch Instances**

cs.CL

IJCAI-ECAI 2022 (the 31st International Joint Conference on  Artificial Intelligence and the 25th European Conference on Artificial  Intelligence)

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00633v1)

**Authors**: Shoujie Tong, Qingxiu Dong, Damai Dai, Yifan song, Tianyu Liu, Baobao Chang, Zhifang Sui

**Abstracts**: Fine-tuning pretrained language models (PLMs) on downstream tasks has become common practice in natural language processing. However, most of the PLMs are vulnerable, e.g., they are brittle under adversarial attacks or imbalanced data, which hinders the application of the PLMs on some downstream tasks, especially in safe-critical scenarios. In this paper, we propose a simple yet effective fine-tuning method called Match-Tuning to force the PLMs to be more robust. For each instance in a batch, we involve other instances in the same batch to interact with it. To be specific, regarding the instances with other labels as a perturbation, Match-Tuning makes the model more robust to noise at the beginning of training. While nearing the end, Match-Tuning focuses more on performing an interpolation among the instances with the same label for better generalization. Extensive experiments on various tasks in GLUE benchmark show that Match-Tuning consistently outperforms the vanilla fine-tuning by $1.64$ scores. Moreover, Match-Tuning exhibits remarkable robustness to adversarial attacks and data imbalance.



## **13. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction**

cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-05-01    [paper-pdf](http://arxiv.org/pdf/2205.01094v1)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.



## **14. Analysis of a blockchain protocol based on LDPC codes**

cs.CR

**SubmitDate**: 2022-04-30    [paper-pdf](http://arxiv.org/pdf/2202.07265v3)

**Authors**: Massimo Battaglioni, Paolo Santini, Giulia Rafaiani, Franco Chiaraluce, Marco Baldi

**Abstracts**: In a blockchain Data Availability Attack (DAA), a malicious node publishes a block header but withholds part of the block, which contains invalid transactions. Honest full nodes, which can download and store the full blockchain, are aware that some data are not available but they have no formal way to prove it to light nodes, i.e., nodes that have limited resources and are not able to access the whole blockchain data. A common solution to counter these attacks exploits linear error correcting codes to encode the block content. A recent protocol, called SPAR, employs coded Merkle trees and low-density parity-check codes to counter DAAs. In this paper, we show that the protocol is less secure than claimed, owing to a redefinition of the adversarial success probability. As a consequence we show that, for some realistic choices of the parameters, the total amount of data downloaded by light nodes is larger than that obtainable with competitor solutions.



## **15. Logically Consistent Adversarial Attacks for Soft Theorem Provers**

cs.LG

IJCAI-ECAI 2022

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2205.00047v1)

**Authors**: Alexander Gaskell, Yishu Miao, Lucia Specia, Francesca Toni

**Abstracts**: Recent efforts within the AI community have yielded impressive results towards "soft theorem proving" over natural language sentences using language models. We propose a novel, generative adversarial framework for probing and improving these models' reasoning capabilities. Adversarial attacks in this domain suffer from the logical inconsistency problem, whereby perturbations to the input may alter the label. Our Logically consistent AdVersarial Attacker, LAVA, addresses this by combining a structured generative process with a symbolic solver, guaranteeing logical consistency. Our framework successfully generates adversarial attacks and identifies global weaknesses common across multiple target models. Our analyses reveal naive heuristics and vulnerabilities in these models' reasoning capabilities, exposing an incomplete grasp of logical deduction under logic programs. Finally, in addition to effective probing of these models, we show that training on the generated samples improves the target model's performance.



## **16. To Trust or Not To Trust Prediction Scores for Membership Inference Attacks**

cs.LG

15 pages, 8 figures, 10 tables

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2111.09076v2)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Kristian Kersting

**Abstracts**: Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures. Consequently, MIAs will miserably fail since overconfidence leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated, and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of models and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions, the more they reveal the training data.



## **17. Adversarial attacks on an optical neural network**

cs.CR

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2205.01226v1)

**Authors**: Shuming Jiao, Ziwei Song, Shuiying Xiang

**Abstracts**: Adversarial attacks have been extensively investigated for machine learning systems including deep learning in the digital domain. However, the adversarial attacks on optical neural networks (ONN) have been seldom considered previously. In this work, we first construct an accurate image classifier with an ONN using a mesh of interconnected Mach-Zehnder interferometers (MZI). Then a corresponding adversarial attack scheme is proposed for the first time. The attacked images are visually very similar to the original ones but the ONN system becomes malfunctioned and generates wrong classification results in most time. The results indicate that adversarial attack is also a significant issue for optical machine learning systems.



## **18. Finding MNEMON: Reviving Memories of Node Embeddings**

cs.LG

To Appear in the 29th ACM Conference on Computer and Communications  Security (CCS), November 7-11, 2022

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.06963v2)

**Authors**: Yun Shen, Yufei Han, Zhikun Zhang, Min Chen, Ting Yu, Michael Backes, Yang Zhang, Gianluca Stringhini

**Abstracts**: Previous security research efforts orbiting around graphs have been exclusively focusing on either (de-)anonymizing the graphs or understanding the security and privacy issues of graph neural networks. Little attention has been paid to understand the privacy risks of integrating the output from graph embedding models (e.g., node embeddings) with complex downstream machine learning pipelines. In this paper, we fill this gap and propose a novel model-agnostic graph recovery attack that exploits the implicit graph structural information preserved in the embeddings of graph nodes. We show that an adversary can recover edges with decent accuracy by only gaining access to the node embedding matrix of the original graph without interactions with the node embedding models. We demonstrate the effectiveness and applicability of our graph recovery attack through extensive experiments.



## **19. Exploration and Exploitation in Federated Learning to Exclude Clients with Poisoned Data**

cs.DC

Accepted at 2022 IWCMC

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.14020v1)

**Authors**: Shadha Tabatabai, Ihab Mohammed, Basheer Qolomany, Abdullatif Albasser, Kashif Ahmad, Mohamed Abdallah, Ala Al-Fuqaha

**Abstracts**: Federated Learning (FL) is one of the hot research topics, and it utilizes Machine Learning (ML) in a distributed manner without directly accessing private data on clients. However, FL faces many challenges, including the difficulty to obtain high accuracy, high communication cost between clients and the server, and security attacks related to adversarial ML. To tackle these three challenges, we propose an FL algorithm inspired by evolutionary techniques. The proposed algorithm groups clients randomly in many clusters, each with a model selected randomly to explore the performance of different models. The clusters are then trained in a repetitive process where the worst performing cluster is removed in each iteration until one cluster remains. In each iteration, some clients are expelled from clusters either due to using poisoned data or low performance. The surviving clients are exploited in the next iteration. The remaining cluster with surviving clients is then used for training the best FL model (i.e., remaining FL model). Communication cost is reduced since fewer clients are used in the final training of the FL model. To evaluate the performance of the proposed algorithm, we conduct a number of experiments using FEMNIST dataset and compare the result against the random FL algorithm. The experimental results show that the proposed algorithm outperforms the baseline algorithm in terms of accuracy, communication cost, and security.



## **20. Backdoor Attacks in Federated Learning by Rare Embeddings and Gradient Ensembling**

cs.LG

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.14017v1)

**Authors**: KiYoon Yoo, Nojun Kwak

**Abstracts**: Recent advances in federated learning have demonstrated its promising capability to learn on decentralized datasets. However, a considerable amount of work has raised concerns due to the potential risks of adversaries participating in the framework to poison the global model for an adversarial purpose. This paper investigates the feasibility of model poisoning for backdoor attacks through \textit{rare word embeddings of NLP models} in text classification and sequence-to-sequence tasks. In text classification, less than 1\% of adversary clients suffices to manipulate the model output without any drop in the performance of clean sentences. For a less complex dataset, a mere 0.1\% of adversary clients is enough to poison the global model effectively. We also propose a technique specialized in the federated learning scheme called gradient ensemble, which enhances the backdoor performance in all experimental settings.



## **21. Using 3D Shadows to Detect Object Hiding Attacks on Autonomous Vehicle Perception**

cs.CV

To appear in the Proceedings of the 2022 IEEE Security and Privacy  Workshop on the Internet of Safe Things (SafeThings 2022)

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.13973v1)

**Authors**: Zhongyuan Hau, Soteris Demetriou, Emil C. Lupu

**Abstracts**: Autonomous Vehicles (AVs) are mostly reliant on LiDAR sensors which enable spatial perception of their surroundings and help make driving decisions. Recent works demonstrated attacks that aim to hide objects from AV perception, which can result in severe consequences. 3D shadows, are regions void of measurements in 3D point clouds which arise from occlusions of objects in a scene. 3D shadows were proposed as a physical invariant valuable for detecting spoofed or fake objects. In this work, we leverage 3D shadows to locate obstacles that are hidden from object detectors. We achieve this by searching for void regions and locating the obstacles that cause these shadows. Our proposed methodology can be used to detect an object that has been hidden by an adversary as these objects, while hidden from 3D object detectors, still induce shadow artifacts in 3D point clouds, which we use for obstacle detection. We show that using 3D shadows for obstacle detection can achieve high accuracy in matching shadows to their object and provide precise prediction of an obstacle's distance from the ego-vehicle.



## **22. Detecting Textual Adversarial Examples Based on Distributional Characteristics of Data Representations**

cs.CL

13 pages, RepL4NLP 2022

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.13853v1)

**Authors**: Na Liu, Mark Dras, Wei Emma Zhang

**Abstracts**: Although deep neural networks have achieved state-of-the-art performance in various machine learning tasks, adversarial examples, constructed by adding small non-random perturbations to correctly classified inputs, successfully fool highly expressive deep classifiers into incorrect predictions. Approaches to adversarial attacks in natural language tasks have boomed in the last five years using character-level, word-level, phrase-level, or sentence-level textual perturbations. While there is some work in NLP on defending against such attacks through proactive methods, like adversarial training, there is to our knowledge no effective general reactive approaches to defence via detection of textual adversarial examples such as is found in the image processing literature. In this paper, we propose two new reactive methods for NLP to fill this gap, which unlike the few limited application baselines from NLP are based entirely on distribution characteristics of learned representations: we adapt one from the image processing literature (Local Intrinsic Dimensionality (LID)), and propose a novel one (MultiDistance Representation Ensemble Method (MDRE)). Adapted LID and MDRE obtain state-of-the-art results on character-level, word-level, and phrase-level attacks on the IMDB dataset as well as on the later two with respect to the MultiNLI dataset. For future research, we publish our code.



## **23. DeepAdversaries: Examining the Robustness of Deep Learning Models for Galaxy Morphology Classification**

cs.LG

19 pages, 7 figures, 5 tables

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2112.14299v2)

**Authors**: Aleksandra Ćiprijanović, Diana Kafkes, Gregory Snyder, F. Javier Sánchez, Gabriel Nathan Perdue, Kevin Pedro, Brian Nord, Sandeep Madireddy, Stefan M. Wild

**Abstracts**: Data processing and analysis pipelines in cosmological survey experiments introduce data perturbations that can significantly degrade the performance of deep learning-based models. Given the increased adoption of supervised deep learning methods for processing and analysis of cosmological survey data, the assessment of data perturbation effects and the development of methods that increase model robustness are increasingly important. In the context of morphological classification of galaxies, we study the effects of perturbations in imaging data. In particular, we examine the consequences of using neural networks when training on baseline data and testing on perturbed data. We consider perturbations associated with two primary sources: 1) increased observational noise as represented by higher levels of Poisson noise and 2) data processing noise incurred by steps such as image compression or telescope errors as represented by one-pixel adversarial attacks. We also test the efficacy of domain adaptation techniques in mitigating the perturbation-driven errors. We use classification accuracy, latent space visualizations, and latent space distance to assess model robustness. Without domain adaptation, we find that processing pixel-level errors easily flip the classification into an incorrect class and that higher observational noise makes the model trained on low-noise data unable to classify galaxy morphologies. On the other hand, we show that training with domain adaptation improves model robustness and mitigates the effects of these perturbations, improving the classification accuracy by 23% on data with higher observational noise. Domain adaptation also increases by a factor of ~2.3 the latent space distance between the baseline and the incorrectly classified one-pixel perturbed image, making the model more robust to inadvertent perturbations.



## **24. Survey and Taxonomy of Adversarial Reconnaissance Techniques**

cs.CR

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2105.04749v2)

**Authors**: Shanto Roy, Nazia Sharmin, Jaime C. Acosta, Christopher Kiekintveld, Aron Laszka

**Abstracts**: Adversaries are often able to penetrate networks and compromise systems by exploiting vulnerabilities in people and systems. The key to the success of these attacks is information that adversaries collect throughout the phases of the cyber kill chain. We summarize and analyze the methods, tactics, and tools that adversaries use to conduct reconnaissance activities throughout the attack process. First, we discuss what types of information adversaries seek, and how and when they can obtain this information. Then, we provide a taxonomy and detailed overview of adversarial reconnaissance techniques. The taxonomy introduces a categorization of reconnaissance techniques based on the source as third-party, human-, and system-based information gathering. This paper provides a comprehensive view of adversarial reconnaissance that can help in understanding and modeling this complex but vital aspect of cyber attacks as well as insights that can improve defensive strategies, such as cyber deception.



## **25. AGIC: Approximate Gradient Inversion Attack on Federated Learning**

cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13784v1)

**Authors**: Jin Xu, Chi Hong, Jiyue Huang, Lydia Y. Chen, Jérémie Decouchant

**Abstracts**: Federated learning is a private-by-design distributed learning paradigm where clients train local models on their own data before a central server aggregates their local updates to compute a global model. Depending on the aggregation method used, the local updates are either the gradients or the weights of local learning models. Recent reconstruction attacks apply a gradient inversion optimization on the gradient update of a single minibatch to reconstruct the private data used by clients during training. As the state-of-the-art reconstruction attacks solely focus on single update, realistic adversarial scenarios are overlooked, such as observation across multiple updates and updates trained from multiple mini-batches. A few studies consider a more challenging adversarial scenario where only model updates based on multiple mini-batches are observable, and resort to computationally expensive simulation to untangle the underlying samples for each local step. In this paper, we propose AGIC, a novel Approximate Gradient Inversion Attack that efficiently and effectively reconstructs images from both model or gradient updates, and across multiple epochs. In a nutshell, AGIC (i) approximates gradient updates of used training samples from model updates to avoid costly simulation procedures, (ii) leverages gradient/model updates collected from multiple epochs, and (iii) assigns increasing weights to layers with respect to the neural network structure for reconstruction quality. We extensively evaluate AGIC on three datasets, CIFAR-10, CIFAR-100 and ImageNet. Our results show that AGIC increases the peak signal-to-noise ratio (PSNR) by up to 50% compared to two representative state-of-the-art gradient inversion attacks. Furthermore, AGIC is faster than the state-of-the-art simulation based attack, e.g., it is 5x faster when attacking FedAvg with 8 local steps in between model updates.



## **26. Formulating Robustness Against Unforeseen Attacks**

cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13779v1)

**Authors**: Sihui Dai, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: Existing defenses against adversarial examples such as adversarial training typically assume that the adversary will conform to a specific or known threat model, such as $\ell_p$ perturbations within a fixed budget. In this paper, we focus on the scenario where there is a mismatch in the threat model assumed by the defense during training, and the actual capabilities of the adversary at test time. We ask the question: if the learner trains against a specific "source" threat model, when can we expect robustness to generalize to a stronger unknown "target" threat model during test-time? Our key contribution is to formally define the problem of learning and generalization with an unforeseen adversary, which helps us reason about the increase in adversarial risk from the conventional perspective of a known adversary. Applying our framework, we derive a generalization bound which relates the generalization gap between source and target threat models to variation of the feature extractor, which measures the expected maximum difference between extracted features across a given threat model. Based on our generalization bound, we propose adversarial training with variation regularization (AT-VR) which reduces variation of the feature extractor across the source threat model during training. We empirically demonstrate that AT-VR can lead to improved generalization to unforeseen attacks during test-time compared to standard adversarial training on Gaussian and image datasets.



## **27. UNBUS: Uncertainty-aware Deep Botnet Detection System in Presence of Perturbed Samples**

cs.CR

8 pages, 5 figures, 5 Tables

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.09502v2)

**Authors**: Rahim Taheri

**Abstracts**: A rising number of botnet families have been successfully detected using deep learning architectures. While the variety of attacks increases, these architectures should become more robust against attacks. They have been proven to be very sensitive to small but well constructed perturbations in the input. Botnet detection requires extremely low false-positive rates (FPR), which are not commonly attainable in contemporary deep learning. Attackers try to increase the FPRs by making poisoned samples. The majority of recent research has focused on the use of model loss functions to build adversarial examples and robust models. In this paper, two LSTM-based classification algorithms for botnet classification with an accuracy higher than 98% are presented. Then, the adversarial attack is proposed, which reduces the accuracy to about 30%. Then, by examining the methods for computing the uncertainty, the defense method is proposed to increase the accuracy to about 70%. By using the deep ensemble and stochastic weight averaging quantification methods it has been investigated the uncertainty of the accuracy in the proposed methods.



## **28. Deepfake Forensics via An Adversarial Game**

cs.CV

Accepted by IEEE Transactions on Image Processing; 13 pages, 4  figures

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2103.13567v2)

**Authors**: Zhi Wang, Yiwen Guo, Wangmeng Zuo

**Abstracts**: With the progress in AI-based facial forgery (i.e., deepfake), people are increasingly concerned about its abuse. Albeit effort has been made for training classification (also known as deepfake detection) models to recognize such forgeries, existing models suffer from poor generalization to unseen forgery technologies and high sensitivity to changes in image/video quality. In this paper, we advocate adversarial training for improving the generalization ability to both unseen facial forgeries and unseen image/video qualities. We believe training with samples that are adversarially crafted to attack the classification models improves the generalization ability considerably. Considering that AI-based face manipulation often leads to high-frequency artifacts that can be easily spotted by models yet difficult to generalize, we further propose a new adversarial training method that attempts to blur out these specific artifacts, by introducing pixel-wise Gaussian blurring models. With adversarial training, the classification models are forced to learn more discriminative and generalizable features, and the effectiveness of our method can be verified by plenty of empirical evidence. Our code will be made publicly available.



## **29. Randomized Smoothing under Attack: How Good is it in Pratice?**

cs.CR

ICASSP 2022

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.14187v1)

**Authors**: Thibault Maho, Teddy Furon, Erwan Le Merrer

**Abstracts**: Randomized smoothing is a recent and celebrated solution to certify the robustness of any classifier. While it indeed provides a theoretical robustness against adversarial attacks, the dimensionality of current classifiers necessarily imposes Monte Carlo approaches for its application in practice. This paper questions the effectiveness of randomized smoothing as a defense, against state of the art black-box attacks. This is a novel perspective, as previous research works considered the certification as an unquestionable guarantee. We first formally highlight the mismatch between a theoretical certification and the practice of attacks on classifiers. We then perform attacks on randomized smoothing as a defense. Our main observation is that there is a major mismatch in the settings of the RS for obtaining high certified robustness or when defeating black box attacks while preserving the classifier accuracy.



## **30. Adversarial Fine-tune with Dynamically Regulated Adversary**

cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13232v1)

**Authors**: Pengyue Hou, Ming Zhou, Jie Han, Petr Musilek, Xingyu Li

**Abstracts**: Adversarial training is an effective method to boost model robustness to malicious, adversarial attacks. However, such improvement in model robustness often leads to a significant sacrifice of standard performance on clean images. In many real-world applications such as health diagnosis and autonomous surgical robotics, the standard performance is more valued over model robustness against such extremely malicious attacks. This leads to the question: To what extent we can boost model robustness without sacrificing standard performance? This work tackles this problem and proposes a simple yet effective transfer learning-based adversarial training strategy that disentangles the negative effects of adversarial samples on model's standard performance. In addition, we introduce a training-friendly adversarial attack algorithm, which facilitates the boost of adversarial robustness without introducing significant training complexity. Extensive experimentation indicates that the proposed method outperforms previous adversarial training algorithms towards the target: to improve model robustness while preserving model's standard performance on clean data.



## **31. An Adversarial Attack Analysis on Malicious Advertisement URL Detection Framework**

cs.LG

13

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13172v1)

**Authors**: Ehsan Nowroozi, Abhishek, Mohammadreza Mohammadi, Mauro Conti

**Abstracts**: Malicious advertisement URLs pose a security risk since they are the source of cyber-attacks, and the need to address this issue is growing in both industry and academia. Generally, the attacker delivers an attack vector to the user by means of an email, an advertisement link or any other means of communication and directs them to a malicious website to steal sensitive information and to defraud them. Existing malicious URL detection techniques are limited and to handle unseen features as well as generalize to test data. In this study, we extract a novel set of lexical and web-scrapped features and employ machine learning technique to set up system for fraudulent advertisement URLs detection. The combination set of six different kinds of features precisely overcome the obfuscation in fraudulent URL classification. Based on different statistical properties, we use twelve different formatted datasets for detection, prediction and classification task. We extend our prediction analysis for mismatched and unlabelled datasets. For this framework, we analyze the performance of four machine learning techniques: Random Forest, Gradient Boost, XGBoost and AdaBoost in the detection part. With our proposed method, we can achieve a false negative rate as low as 0.0037 while maintaining high accuracy of 99.63%. Moreover, we devise a novel unsupervised technique for data clustering using K- Means algorithm for the visual analysis. This paper analyses the vulnerability of decision tree-based models using the limited knowledge attack scenario. We considered the exploratory attack and implemented Zeroth Order Optimization adversarial attack on the detection models.



## **32. SSR-GNNs: Stroke-based Sketch Representation with Graph Neural Networks**

cs.CV

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13153v1)

**Authors**: Sheng Cheng, Yi Ren, Yezhou Yang

**Abstracts**: This paper follows cognitive studies to investigate a graph representation for sketches, where the information of strokes, i.e., parts of a sketch, are encoded on vertices and information of inter-stroke on edges. The resultant graph representation facilitates the training of a Graph Neural Networks for classification tasks, and achieves accuracy and robustness comparable to the state-of-the-art against translation and rotation attacks, as well as stronger attacks on graph vertices and topologies, i.e., modifications and addition of strokes, all without resorting to adversarial training. Prior studies on sketches, e.g., graph transformers, encode control points of stroke on vertices, which are not invariant to spatial transformations. In contrary, we encode vertices and edges using pairwise distances among control points to achieve invariance. Compared with existing generative sketch model for one-shot classification, our method does not rely on run-time statistical inference. Lastly, the proposed representation enables generation of novel sketches that are structurally similar to while separable from the existing dataset.



## **33. Defending Against Person Hiding Adversarial Patch Attack with a Universal White Frame**

cs.CV

Submitted by NeurIPS 2021 with response letter to the anonymous  reviewers' comments

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13004v1)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstracts**: Object detection has attracted great attention in the computer vision area and has emerged as an indispensable component in many vision systems. In the era of deep learning, many high-performance object detection networks have been proposed. Although these detection networks show high performance, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the detection network in the physical world. In particular, person-hiding attacks are emerging as a serious problem in many safety-critical applications such as autonomous driving and surveillance systems. Although it is necessary to defend against an adversarial patch attack, very few efforts have been dedicated to defending against person-hiding attacks. To tackle the problem, in this paper, we propose a novel defense strategy that mitigates a person-hiding attack by optimizing defense patterns, while previous methods optimize the model. In the proposed method, a frame-shaped pattern called a 'universal white frame' (UWF) is optimized and placed on the outside of the image. To defend against adversarial patch attacks, UWF should have three properties (i) suppressing the effect of the adversarial patch, (ii) maintaining its original prediction, and (iii) applicable regardless of images. To satisfy the aforementioned properties, we propose a novel pattern optimization algorithm that can defend against the adversarial patch. Through comprehensive experiments, we demonstrate that the proposed method effectively defends against the adversarial patch attack.



## **34. The MeVer DeepFake Detection Service: Lessons Learnt from Developing and Deploying in the Wild**

cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.12816v1)

**Authors**: Spyridon Baxevanakis, Giorgos Kordopatis-Zilos, Panagiotis Galopoulos, Lazaros Apostolidis, Killian Levacher, Ipek B. Schlicht, Denis Teyssou, Ioannis Kompatsiaris, Symeon Papadopoulos

**Abstracts**: Enabled by recent improvements in generation methodologies, DeepFakes have become mainstream due to their increasingly better visual quality, the increase in easy-to-use generation tools and the rapid dissemination through social media. This fact poses a severe threat to our societies with the potential to erode social cohesion and influence our democracies. To mitigate the threat, numerous DeepFake detection schemes have been introduced in the literature but very few provide a web service that can be used in the wild. In this paper, we introduce the MeVer DeepFake detection service, a web service detecting deep learning manipulations in images and video. We present the design and implementation of the proposed processing pipeline that involves a model ensemble scheme, and we endow the service with a model card for transparency. Experimental results show that our service performs robustly on the three benchmark datasets while being vulnerable to Adversarial Attacks. Finally, we outline our experience and lessons learned when deploying a research system into production in the hopes that it will be useful to other academic and industry teams.



## **35. Improving the Transferability of Adversarial Examples with Restructure Embedded Patches**

cs.CV

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.12680v1)

**Authors**: Huipeng Zhou, Yu-an Tan, Yajie Wang, Haoran Lyu, Shangbo Wu, Yuanzhang Li

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance in various computer vision tasks. However, the adversarial examples generated by ViTs are challenging to transfer to other networks with different structures. Recent attack methods do not consider the specificity of ViTs architecture and self-attention mechanism, which leads to poor transferability of the generated adversarial samples by ViTs. We attack the unique self-attention mechanism in ViTs by restructuring the embedded patches of the input. The restructured embedded patches enable the self-attention mechanism to obtain more diverse patches connections and help ViTs keep regions of interest on the object. Therefore, we propose an attack method against the unique self-attention mechanism in ViTs, called Self-Attention Patches Restructure (SAPR). Our method is simple to implement yet efficient and applicable to any self-attention based network and gradient transferability-based attack methods. We evaluate attack transferability on black-box models with different structures. The result show that our method generates adversarial examples on white-box ViTs with higher transferability and higher image quality. Our research advances the development of black-box transfer attacks on ViTs and demonstrates the feasibility of using white-box ViTs to attack other black-box models.



## **36. Data Bootstrapping Approaches to Improve Low Resource Abusive Language Detection for Indic Languages**

cs.CL

Accepted at HT '22: 33rd ACM Conference on Hypertext and Social Media

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12543v1)

**Authors**: Mithun Das, Somnath Banerjee, Animesh Mukherjee

**Abstracts**: Abusive language is a growing concern in many social media platforms. Repeated exposure to abusive speech has created physiological effects on the target users. Thus, the problem of abusive language should be addressed in all forms for online peace and safety. While extensive research exists in abusive speech detection, most studies focus on English. Recently, many smearing incidents have occurred in India, which provoked diverse forms of abusive speech in online space in various languages based on the geographic location. Therefore it is essential to deal with such malicious content. In this paper, to bridge the gap, we demonstrate a large-scale analysis of multilingual abusive speech in Indic languages. We examine different interlingual transfer mechanisms and observe the performance of various multilingual models for abusive speech detection for eight different Indic languages. We also experiment to show how robust these models are on adversarial attacks. Finally, we conduct an in-depth error analysis by looking into the models' misclassified posts across various settings. We have made our code and models public for other researchers.



## **37. Restricted Black-box Adversarial Attack Against DeepFake Face Swapping**

cs.CV

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12347v1)

**Authors**: Junhao Dong, Yuan Wang, Jianhuang Lai, Xiaohua Xie

**Abstracts**: DeepFake face swapping presents a significant threat to online security and social media, which can replace the source face in an arbitrary photo/video with the target face of an entirely different person. In order to prevent this fraud, some researchers have begun to study the adversarial methods against DeepFake or face manipulation. However, existing works focus on the white-box setting or the black-box setting driven by abundant queries, which severely limits the practical application of these methods. To tackle this problem, we introduce a practical adversarial attack that does not require any queries to the facial image forgery model. Our method is built on a substitute model persuing for face reconstruction and then transfers adversarial examples from the substitute model directly to inaccessible black-box DeepFake models. Specially, we propose the Transferable Cycle Adversary Generative Adversarial Network (TCA-GAN) to construct the adversarial perturbation for disrupting unknown DeepFake systems. We also present a novel post-regularization module for enhancing the transferability of generated adversarial examples. To comprehensively measure the effectiveness of our approaches, we construct a challenging benchmark of DeepFake adversarial attacks for future development. Extensive experiments impressively show that the proposed adversarial attack method makes the visual quality of DeepFake face images plummet so that they are easier to be detected by humans and algorithms. Moreover, we demonstrate that the proposed algorithm can be generalized to offer face image protection against various face translation methods.



## **38. Boosting Adversarial Transferability of MLP-Mixer**

cs.CV

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12204v1)

**Authors**: Haoran Lyu, Yajie Wang, Yu-an Tan, Huipeng Zhou, Yuhang Zhao, Quanxin Zhang

**Abstracts**: The security of models based on new architectures such as MLP-Mixer and ViTs needs to be studied urgently. However, most of the current researches are mainly aimed at the adversarial attack against ViTs, and there is still relatively little adversarial work on MLP-mixer. We propose an adversarial attack method against MLP-Mixer called Maxwell's demon Attack (MA). MA breaks the channel-mixing and token-mixing mechanism of MLP-Mixer by controlling the part input of MLP-Mixer's each Mixer layer, and disturbs MLP-Mixer to obtain the main information of images. Our method can mask the part input of the Mixer layer, avoid overfitting of the adversarial examples to the source model, and improve the transferability of cross-architecture. Extensive experimental evaluation demonstrates the effectiveness and superior performance of the proposed MA. Our method can be easily combined with existing methods and can improve the transferability by up to 38.0% on MLP-based ResMLP. Adversarial examples produced by our method on MLP-Mixer are able to exceed the transferability of adversarial examples produced using DenseNet against CNNs. To the best of our knowledge, we are the first work to study adversarial transferability of MLP-Mixer.



## **39. Mixed Strategies for Security Games with General Defending Requirements**

cs.GT

Accepted by IJCAI-2022

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12158v1)

**Authors**: Rufan Bai, Haoxing Lin, Xinyu Yang, Xiaowei Wu, Minming Li, Weijia Jia

**Abstracts**: The Stackelberg security game is played between a defender and an attacker, where the defender needs to allocate a limited amount of resources to multiple targets in order to minimize the loss due to adversarial attack by the attacker. While allowing targets to have different values, classic settings often assume uniform requirements to defend the targets. This enables existing results that study mixed strategies (randomized allocation algorithms) to adopt a compact representation of the mixed strategies.   In this work, we initiate the study of mixed strategies for the security games in which the targets can have different defending requirements. In contrast to the case of uniform defending requirement, for which an optimal mixed strategy can be computed efficiently, we show that computing the optimal mixed strategy is NP-hard for the general defending requirements setting. However, we show that strong upper and lower bounds for the optimal mixed strategy defending result can be derived. We propose an efficient close-to-optimal Patching algorithm that computes mixed strategies that use only few pure strategies. We also study the setting when the game is played on a network and resource sharing is enabled between neighboring targets. Our experimental results demonstrate the effectiveness of our algorithm in several large real-world datasets.



## **40. Source-independent quantum random number generator against detector blinding attacks**

quant-ph

14 pages, 7 figures, 6 tables, comments are welcome

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12156v1)

**Authors**: Wen-Bo Liu, Yu-Shuo Lu, Yao Fu, Si-Cheng Huang, Ze-Jie Yin, Kun Jiang, Hua-Lei Yin, Zeng-Bing Chen

**Abstracts**: Randomness, mainly in the form of random numbers, is the fundamental prerequisite for the security of many cryptographic tasks. Quantum randomness can be extracted even if adversaries are fully aware of the protocol and even control the randomness source. However, an adversary can further manipulate the randomness via detector blinding attacks, which are a hacking attack suffered by protocols with trusted detectors. Here, by treating no-click events as valid error events, we propose a quantum random number generation protocol that can simultaneously address source vulnerability and ferocious detector blinding attacks. The method can be extended to high-dimensional random number generation. We experimentally demonstrate the ability of our protocol to generate random numbers for two-dimensional measurement with a generation speed of 0.515 Mbps, which is two orders of magnitude higher than that of device-independent protocols that can address both issues of imperfect sources and imperfect detectors.



## **41. Self-recoverable Adversarial Examples: A New Effective Protection Mechanism in Social Networks**

cs.CV

13 pages, 11 figures

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12050v1)

**Authors**: Jiawei Zhang, Jinwei Wang, Hao Wang, Xiangyang Luo

**Abstracts**: Malicious intelligent algorithms greatly threaten the security of social users' privacy by detecting and analyzing the uploaded photos to social network platforms. The destruction to DNNs brought by the adversarial attack sparks the potential that adversarial examples serve as a new protection mechanism for privacy security in social networks. However, the existing adversarial example does not have recoverability for serving as an effective protection mechanism. To address this issue, we propose a recoverable generative adversarial network to generate self-recoverable adversarial examples. By modeling the adversarial attack and recovery as a united task, our method can minimize the error of the recovered examples while maximizing the attack ability, resulting in better recoverability of adversarial examples. To further boost the recoverability of these examples, we exploit a dimension reducer to optimize the distribution of adversarial perturbation. The experimental results prove that the adversarial examples generated by the proposed method present superior recoverability, attack ability, and robustness on different datasets and network architectures, which ensure its effectiveness as a protection mechanism in social networks.



## **42. Can Rationalization Improve Robustness?**

cs.CL

Accepted to NAACL 2022

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11790v1)

**Authors**: Howard Chen, Jacqueline He, Karthik Narasimhan, Danqi Chen

**Abstracts**: A growing line of work has investigated the development of neural NLP models that can produce rationales--subsets of input that can explain their model predictions. In this paper, we ask whether such rationale models can also provide robustness to adversarial attacks in addition to their interpretable nature. Since these models need to first generate rationales ("rationalizer") before making predictions ("predictor"), they have the potential to ignore noise or adversarially added text by simply masking it out of the generated rationale. To this end, we systematically generate various types of 'AddText' attacks for both token and sentence-level rationalization tasks, and perform an extensive empirical evaluation of state-of-the-art rationale models across five different tasks. Our experiments reveal that the rationale models show the promise to improve robustness, while they struggle in certain scenarios--when the rationalizer is sensitive to positional bias or lexical choices of attack text. Further, leveraging human rationale as supervision does not always translate to better performance. Our study is a first step towards exploring the interplay between interpretability and robustness in the rationalize-then-predict framework.



## **43. Discovering Exfiltration Paths Using Reinforcement Learning with Attack Graphs**

cs.CR

The 5th IEEE Conference on Dependable and Secure Computing (IEEE DSC  2022)

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.12416v2)

**Authors**: Tyler Cody, Abdul Rahman, Christopher Redino, Lanxiao Huang, Ryan Clark, Akshay Kakkar, Deepak Kushwaha, Paul Park, Peter Beling, Edward Bowen

**Abstracts**: Reinforcement learning (RL), in conjunction with attack graphs and cyber terrain, are used to develop reward and state associated with determination of optimal paths for exfiltration of data in enterprise networks. This work builds on previous crown jewels (CJ) identification that focused on the target goal of computing optimal paths that adversaries may traverse toward compromising CJs or hosts within their proximity. This work inverts the previous CJ approach based on the assumption that data has been stolen and now must be quietly exfiltrated from the network. RL is utilized to support the development of a reward function based on the identification of those paths where adversaries desire reduced detection. Results demonstrate promising performance for a sizable network environment.



## **44. Reconstructing Training Data with Informed Adversaries**

cs.CR

Published at "2022 IEEE Symposium on Security and Privacy (SP)"

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.04845v2)

**Authors**: Borja Balle, Giovanni Cherubin, Jamie Hayes

**Abstracts**: Given access to a machine learning model, can an adversary reconstruct the model's training data? This work studies this question from the lens of a powerful informed adversary who knows all the training data points except one. By instantiating concrete attacks, we show it is feasible to reconstruct the remaining data point in this stringent threat model. For convex models (e.g. logistic regression), reconstruction attacks are simple and can be derived in closed-form. For more general models (e.g. neural networks), we propose an attack strategy based on training a reconstructor network that receives as input the weights of the model under attack and produces as output the target data point. We demonstrate the effectiveness of our attack on image classifiers trained on MNIST and CIFAR-10, and systematically investigate which factors of standard machine learning pipelines affect reconstruction success. Finally, we theoretically investigate what amount of differential privacy suffices to mitigate reconstruction attacks by informed adversaries. Our work provides an effective reconstruction attack that model developers can use to assess memorization of individual points in general settings beyond those considered in previous works (e.g. generative language models or access to training gradients); it shows that standard models have the capacity to store enough information to enable high-fidelity reconstruction of training data points; and it demonstrates that differential privacy can successfully mitigate such attacks in a parameter regime where utility degradation is minimal.



## **45. A Simple Structure For Building A Robust Model**

cs.CV

10 pages, 3 figures, 4 tables

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11596v1)

**Authors**: Xiao Tan, JingBo Gao, Ruolin Li

**Abstracts**: As deep learning applications, especially programs of computer vision, are increasingly deployed in our lives, we have to think more urgently about the security of these applications.One effective way to improve the security of deep learning models is to perform adversarial training, which allows the model to be compatible with samples that are deliberately created for use in attacking the model.Based on this, we propose a simple architecture to build a model with a certain degree of robustness, which improves the robustness of the trained network by adding an adversarial sample detection network for cooperative training.At the same time, we design a new data sampling strategy that incorporates multiple existing attacks, allowing the model to adapt to many different adversarial attacks with a single training.We conducted some experiments to test the effectiveness of this design based on Cifar10 dataset, and the results indicate that it has some degree of positive effect on the robustness of the model.Our code could be found at https://github.com/dowdyboy/simple_structure_for_robust_model.



## **46. Dominating Vertical Collaborative Learning Systems**

cs.CR

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.02775v2)

**Authors**: Qi Pang, Yuanyuan Yuan, Shuai Wang

**Abstracts**: Vertical collaborative learning system also known as vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-preserving manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individual organizations.   Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in collaborative learning scenarios.   We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods to synthesize ADIs of various formats and exploit common VFL systems. We further launch greybox fuzz testing, guided by the resiliency score of "victim" participants, to perturb adversary-controlled inputs and systematically explore the VFL attack surface in a privacy-preserving manner. We conduct an in-depth study on the influence of critical parameters and settings in synthesizing ADIs. Our study reveals new VFL attack opportunities, promoting the identification of unknown threats before breaches and building more secure VFL systems.



## **47. Real or Virtual: A Video Conferencing Background Manipulation-Detection System**

cs.CV

34 pages. arXiv admin note: text overlap with arXiv:2106.15130

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11853v1)

**Authors**: Ehsan Nowroozi, Yassine Mekdad, Mauro Conti, Simone Milani, Selcuk Uluagac, Berrin Yanikoglu

**Abstracts**: Recently, the popularity and wide use of the last-generation video conferencing technologies created an exponential growth in its market size. Such technology allows participants in different geographic regions to have a virtual face-to-face meeting. Additionally, it enables users to employ a virtual background to conceal their own environment due to privacy concerns or to reduce distractions, particularly in professional settings. Nevertheless, in scenarios where the users should not hide their actual locations, they may mislead other participants by claiming their virtual background as a real one. Therefore, it is crucial to develop tools and strategies to detect the authenticity of the considered virtual background. In this paper, we present a detection strategy to distinguish between real and virtual video conferencing user backgrounds. We demonstrate that our detector is robust against two attack scenarios. The first scenario considers the case where the detector is unaware about the attacks and inn the second scenario, we make the detector aware of the adversarial attacks, which we refer to Adversarial Multimedia Forensics (i.e, the forensically-edited frames are included in the training set). Given the lack of publicly available dataset of virtual and real backgrounds for video conferencing, we created our own dataset and made them publicly available [1]. Then, we demonstrate the robustness of our detector against different adversarial attacks that the adversary considers. Ultimately, our detector's performance is significant against the CRSPAM1372 [2] features, and post-processing operations such as geometric transformations with different quality factors that the attacker may choose. Moreover, our performance results shows that we can perfectly identify a real from a virtual background with an accuracy of 99.80%.



## **48. A Hybrid Defense Method against Adversarial Attacks on Traffic Sign Classifiers in Autonomous Vehicles**

cs.CR

13 pages, 8 figures

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2205.01225v1)

**Authors**: Zadid Khan, Mashrur Chowdhury, Sakib Mahmud Khan

**Abstracts**: Adversarial attacks can make deep neural network (DNN) models predict incorrect output labels, such as misclassified traffic signs, for autonomous vehicle (AV) perception modules. Resilience against adversarial attacks can help AVs navigate safely on the road by avoiding misclassication of signs or objects. This DNN-based study develops a resilient traffic sign classifier for AVs that uses a hybrid defense method. We use transfer learning to retrain the Inception-V3 and Resnet-152 models as traffic sign classifiers. This method also utilizes a combination of three different strategies: random filtering, ensembling, and local feature mapping. We use the random cropping and resizing technique for random filtering, plurality voting as ensembling strategy and an optical character recognition model as a local feature mapper. This DNN-based hybrid defense method has been tested for the no attack scenario and against well-known untargeted adversarial attacks (e.g., Projected Gradient Descent or PGD, Fast Gradient Sign Method or FGSM, Momentum Iterative Method or MIM attack, and Carlini and Wagner or C&W). We find that our hybrid defense method achieves 99% average traffic sign classification accuracy for the no attack scenario and 88% average traffic sign classification accuracy for all attack scenarios. Moreover, the hybrid defense method, presented in this study, improves the accuracy for traffic sign classification compared to the traditional defense methods (i.e., JPEG filtering, feature squeezing, binary filtering, and random filtering) up to 6%, 50%, and 55% for FGSM, MIM, and PGD attacks, respectively.



## **49. Improving Deep Learning Model Robustness Against Adversarial Attack by Increasing the Network Capacity**

cs.LG

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11357v1)

**Authors**: Marco Marchetti, Edmond S. L. Ho

**Abstracts**: Nowadays, we are more and more reliant on Deep Learning (DL) models and thus it is essential to safeguard the security of these systems. This paper explores the security issues in Deep Learning and analyses, through the use of experiments, the way forward to build more resilient models. Experiments are conducted to identify the strengths and weaknesses of a new approach to improve the robustness of DL models against adversarial attacks. The results show improvements and new ideas that can be used as recommendations for researchers and practitioners to create increasingly better DL algorithms.



## **50. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

cs.CR

10 pages, 8 figures

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11307v1)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstracts**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.



