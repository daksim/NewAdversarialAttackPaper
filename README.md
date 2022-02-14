# Latest Adversarial Attack Papers
**update at 2022-02-15 06:31:45**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Are socially-aware trajectory prediction models really socially-aware?**

cs.CV

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2108.10879v2)

**Authors**: Saeed Saadatnejad, Mohammadhossein Bahari, Pedram Khorsandi, Mohammad Saneian, Seyed-Mohsen Moosavi-Dezfooli, Alexandre Alahi

**Abstracts**: Our field has recently witnessed an arms race of neural network-based trajectory predictors. While these predictors are at the core of many applications such as autonomous navigation or pedestrian flow simulations, their adversarial robustness has not been carefully studied. In this paper, we introduce a socially-attended attack to assess the social understanding of prediction models in terms of collision avoidance. An attack is a small yet carefully-crafted perturbations to fail predictors. Technically, we define collision as a failure mode of the output, and propose hard- and soft-attention mechanisms to guide our attack. Thanks to our attack, we shed light on the limitations of the current models in terms of their social understanding. We demonstrate the strengths of our method on the recent trajectory prediction models. Finally, we show that our attack can be employed to increase the social understanding of state-of-the-art models. The code is available online: https://s-attack.github.io/



## **2. White-Box Attacks on Hate-speech BERT Classifiers in German with Explicit and Implicit Character Level Defense**

cs.CL

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05778v1)

**Authors**: Shahrukh Khan, Mahnoor Shahid, Navdeeppal Singh

**Abstracts**: In this work, we evaluate the adversarial robustness of BERT models trained on German Hate Speech datasets. We also complement our evaluation with two novel white-box character and word level attacks thereby contributing to the range of attacks available. Furthermore, we also perform a comparison of two novel character-level defense strategies and evaluate their robustness with one another.



## **3. Using Random Perturbations to Mitigate Adversarial Attacks on Sentiment Analysis Models**

cs.CL

To be published in the proceedings for the 18th International  Conference on Natural Language Processing (ICON 2021)

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05758v1)

**Authors**: Abigail Swenor, Jugal Kalita

**Abstracts**: Attacks on deep learning models are often difficult to identify and therefore are difficult to protect against. This problem is exacerbated by the use of public datasets that typically are not manually inspected before use. In this paper, we offer a solution to this vulnerability by using, during testing, random perturbations such as spelling correction if necessary, substitution by random synonym, or simply dropping the word. These perturbations are applied to random words in random sentences to defend NLP models against adversarial attacks. Our Random Perturbations Defense and Increased Randomness Defense methods are successful in returning attacked models to similar accuracy of models before attacks. The original accuracy of the model used in this work is 80% for sentiment classification. After undergoing attacks, the accuracy drops to accuracy between 0% and 44%. After applying our defense methods, the accuracy of the model is returned to the original accuracy within statistical significance.



## **4. On the Detection of Adaptive Adversarial Attacks in Speaker Verification Systems**

cs.CR

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05725v1)

**Authors**: Zesheng Chen

**Abstracts**: Speaker verification systems have been widely used in smart phones and Internet of things devices to identify a legitimate user. In recent work, it has been shown that adversarial attacks, such as FAKEBOB, can work effectively against speaker verification systems. The goal of this paper is to design a detector that can distinguish an original audio from an audio contaminated by adversarial attacks. Specifically, our designed detector, called MEH-FEST, calculates the minimum energy in high frequencies from the short-time Fourier transform of an audio and uses it as a detection metric. Through both analysis and experiments, we show that our proposed detector is easy to implement, fast to process an input audio, and effective in determining whether an audio is corrupted by FAKEBOB attacks. The experimental results indicate that the detector is extremely effective: with near zero false positive and false negative rates for detecting FAKEBOB attacks in Gaussian mixture model (GMM) and i-vector speaker verification systems. Moreover, adaptive adversarial attacks against our proposed detector and their countermeasures are discussed and studied, showing the game between attackers and defenders.



## **5. Towards Adversarially Robust Deepfake Detection: An Ensemble Approach**

cs.LG

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05687v1)

**Authors**: Ashish Hooda, Neal Mangaokar, Ryan Feng, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstracts**: Detecting deepfakes is an important problem, but recent work has shown that DNN-based deepfake detectors are brittle against adversarial deepfakes, in which an adversary adds imperceptible perturbations to a deepfake to evade detection. In this work, we show that a modification to the detection strategy in which we replace a single classifier with a carefully chosen ensemble, in which input transformations for each model in the ensemble induces pairwise orthogonal gradients, can significantly improve robustness beyond the de facto solution of adversarial training. We present theoretical results to show that such orthogonal gradients can help thwart a first-order adversary by reducing the dimensionality of the input subspace in which adversarial deepfakes lie. We validate the results empirically by instantiating and evaluating a randomized version of such "orthogonal" ensembles for adversarial deepfake detection and find that these randomized ensembles exhibit significantly higher robustness as deepfake detectors compared to state-of-the-art deepfake detectors against adversarial deepfakes, even those created using strong PGD-500 attacks.



## **6. FAAG: Fast Adversarial Audio Generation through Interactive Attack Optimisation**

cs.SD

**SubmitDate**: 2022-02-11    [paper-pdf](http://arxiv.org/pdf/2202.05416v1)

**Authors**: Yuantian Miao, Chao Chen, Lei Pan, Jun Zhang, Yang Xiang

**Abstracts**: Automatic Speech Recognition services (ASRs) inherit deep neural networks' vulnerabilities like crafted adversarial examples. Existing methods often suffer from low efficiency because the target phases are added to the entire audio sample, resulting in high demand for computational resources. This paper proposes a novel scheme named FAAG as an iterative optimization-based method to generate targeted adversarial examples quickly. By injecting the noise over the beginning part of the audio, FAAG generates adversarial audio in high quality with a high success rate timely. Specifically, we use audio's logits output to map each character in the transcription to an approximate position of the audio's frame. Thus, an adversarial example can be generated by FAAG in approximately two minutes using CPUs only and around ten seconds with one GPU while maintaining an average success rate over 85%. Specifically, the FAAG method can speed up around 60% compared with the baseline method during the adversarial example generation process. Furthermore, we found that appending benign audio to any suspicious examples can effectively defend against the targeted adversarial attack. We hope that this work paves the way for inventing new adversarial attacks against speech recognition with computational constraints.



## **7. SoK: Certified Robustness for Deep Neural Networks**

cs.LG

14 pages for the main text; recent advances (till Feb 2022) included

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2009.04131v6)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which usually can be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize the certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate over 20 representative certifiably robust approaches for a wide range of DNNs.



## **8. Towards Assessing and Characterizing the Semantic Robustness of Face Recognition**

cs.CV

26 pages, 18 figures

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2202.04978v1)

**Authors**: Juan C. Pérez, Motasem Alfarra, Ali Thabet, Pablo Arbeláez, Bernard Ghanem

**Abstracts**: Deep Neural Networks (DNNs) lack robustness against imperceptible perturbations to their input. Face Recognition Models (FRMs) based on DNNs inherit this vulnerability. We propose a methodology for assessing and characterizing the robustness of FRMs against semantic perturbations to their input. Our methodology causes FRMs to malfunction by designing adversarial attacks that search for identity-preserving modifications to faces. In particular, given a face, our attacks find identity-preserving variants of the face such that an FRM fails to recognize the images belonging to the same identity. We model these identity-preserving semantic modifications via direction- and magnitude-constrained perturbations in the latent space of StyleGAN. We further propose to characterize the semantic robustness of an FRM by statistically describing the perturbations that induce the FRM to malfunction. Finally, we combine our methodology with a certification technique, thus providing (i) theoretical guarantees on the performance of an FRM, and (ii) a formal description of how an FRM may model the notion of face identity.



## **9. Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains**

cs.CV

Accepted by ICLR 2022

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2201.11528v3)

**Authors**: Qilong Zhang, Xiaodan Li, Yuefeng Chen, Jingkuan Song, Lianli Gao, Yuan He, Hui Xue

**Abstracts**: Adversarial examples have posed a severe threat to deep neural networks due to their transferable nature. Currently, various works have paid great efforts to enhance the cross-model transferability, which mostly assume the substitute model is trained in the same domain as the target model. However, in reality, the relevant information of the deployed model is unlikely to leak. Hence, it is vital to build a more practical black-box threat model to overcome this limitation and evaluate the vulnerability of deployed models. In this paper, with only the knowledge of the ImageNet domain, we propose a Beyond ImageNet Attack (BIA) to investigate the transferability towards black-box domains (unknown classification tasks). Specifically, we leverage a generative model to learn the adversarial function for disrupting low-level features of input images. Based on this framework, we further propose two variants to narrow the gap between the source and target domains from the data and model perspectives, respectively. Extensive experiments on coarse-grained and fine-grained domains demonstrate the effectiveness of our proposed methods. Notably, our methods outperform state-of-the-art approaches by up to 7.71\% (towards coarse-grained domains) and 25.91\% (towards fine-grained domains) on average. Our code is available at \url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.



## **10. Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios**

cs.CV

7 pages, 3 figures

**SubmitDate**: 2022-02-10    [paper-pdf](http://arxiv.org/pdf/2202.04781v1)

**Authors**: Jung Im Choi, Qing Tian

**Abstracts**: Visual detection is a key task in autonomous driving, and it serves as one foundation for self-driving planning and control. Deep neural networks have achieved promising results in various computer vision tasks, but they are known to be vulnerable to adversarial attacks. A comprehensive understanding of deep visual detectors' vulnerability is required before people can improve their robustness. However, only a few adversarial attack/defense works have focused on object detection, and most of them employed only classification and/or localization losses, ignoring the objectness aspect. In this paper, we identify a serious objectness-related adversarial vulnerability in YOLO detectors and present an effective attack strategy aiming the objectness aspect of visual detection in autonomous vehicles. Furthermore, to address such vulnerability, we propose a new objectness-aware adversarial training approach for visual detection. Experiments show that the proposed attack targeting the objectness aspect is 45.17% and 43.50% more effective than those generated from classification and/or localization losses on the KITTI and COCO_traffic datasets, respectively. Also, the proposed adversarial defense approach can improve the detectors' robustness against objectness-oriented attacks by up to 21% and 12% mAP on KITTI and COCO_traffic, respectively.



## **11. Layer-wise Regularized Adversarial Training using Layers Sustainability Analysis (LSA) framework**

cs.CV

Layers Sustainability Analysis (LSA) framework

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.02626v2)

**Authors**: Mohammad Khalooei, Mohammad Mehdi Homayounpour, Maryam Amirmazlaghani

**Abstracts**: Deep neural network models are used today in various applications of artificial intelligence, the strengthening of which, in the face of adversarial attacks is of particular importance. An appropriate solution to adversarial attacks is adversarial training, which reaches a trade-off between robustness and generalization. This paper introduces a novel framework (Layer Sustainability Analysis (LSA)) for the analysis of layer vulnerability in a given neural network in the scenario of adversarial attacks. LSA can be a helpful toolkit to assess deep neural networks and to extend adversarial training approaches towards improving the sustainability of model layers via layer monitoring and analysis. The LSA framework identifies a list of Most Vulnerable Layers (MVL list) of a given network. The relative error, as a comparison measure, is used to evaluate the representation sustainability of each layer against adversarial attack inputs. The proposed approach for obtaining robust neural networks to fend off adversarial attacks is based on a layer-wise regularization (LR) over LSA proposal(s) for adversarial training (AT); i.e. the AT-LR procedure. AT-LR could be used with any benchmark adversarial attack to reduce the vulnerability of network layers and to improve conventional adversarial training approaches. The proposed idea performs well theoretically and experimentally for state-of-the-art multilayer perceptron and convolutional neural network architectures. Compared with the AT-LR and its corresponding base adversarial training, the classification accuracy of more significant perturbations increased by 16.35%, 21.79%, and 10.730% on Moon, MNIST, and CIFAR-10 benchmark datasets in comparison with the AT-LR and its corresponding base adversarial training, respectively. The LSA framework is available and published at https://github.com/khalooei/LSA.



## **12. IoTMonitor: A Hidden Markov Model-based Security System to Identify Crucial Attack Nodes in Trigger-action IoT Platforms**

cs.CR

This paper appears in the 2022 IEEE Wireless Communications and  Networking Conference (WCNC 2022). Personal use of this material is  permitted. Permission from IEEE must be obtained for all other uses

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04620v1)

**Authors**: Md Morshed Alam, Md Sajidul Islam Sajid, Weichao Wang, Jinpeng Wei

**Abstracts**: With the emergence and fast development of trigger-action platforms in IoT settings, security vulnerabilities caused by the interactions among IoT devices become more prevalent. The event occurrence at one device triggers an action in another device, which may eventually contribute to the creation of a chain of events in a network. Adversaries exploit the chain effect to compromise IoT devices and trigger actions of interest remotely just by injecting malicious events into the chain. To address security vulnerabilities caused by trigger-action scenarios, existing research efforts focus on the validation of the security properties of devices or verification of the occurrence of certain events based on their physical fingerprints on a device. We propose IoTMonitor, a security analysis system that discerns the underlying chain of event occurrences with the highest probability by observing a chain of physical evidence collected by sensors. We use the Baum-Welch algorithm to estimate transition and emission probabilities and the Viterbi algorithm to discern the event sequence. We can then identify the crucial nodes in the trigger-action sequence whose compromise allows attackers to reach their final goals. The experiment results of our designed system upon the PEEVES datasets show that we can rebuild the event occurrence sequence with high accuracy from the observations and identify the crucial nodes on the attack paths.



## **13. False Memory Formation in Continual Learners Through Imperceptible Backdoor Trigger**

cs.LG

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04479v1)

**Authors**: Muhammad Umer, Robi Polikar

**Abstracts**: In this brief, we show that sequentially learning new information presented to a continual (incremental) learning model introduces new security risks: an intelligent adversary can introduce small amount of misinformation to the model during training to cause deliberate forgetting of a specific task or class at test time, thus creating "false memory" about that task. We demonstrate such an adversary's ability to assume control of the model by injecting "backdoor" attack samples to commonly used generative replay and regularization based continual learning approaches using continual learning benchmark variants of MNIST, as well as the more challenging SVHN and CIFAR 10 datasets. Perhaps most damaging, we show this vulnerability to be very acute and exceptionally effective: the backdoor pattern in our attack model can be imperceptible to human eye, can be provided at any point in time, can be added into the training data of even a single possibly unrelated task and can be achieved with as few as just 1\% of total training dataset of a single task.



## **14. ARIBA: Towards Accurate and Robust Identification of Backdoor Attacks in Federated Learning**

cs.AI

17 pages, 11 figures

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04311v1)

**Authors**: Yuxi Mi, Jihong Guan, Shuigeng Zhou

**Abstracts**: The distributed nature and privacy-preserving characteristics of federated learning make it prone to the threat of poisoning attacks, especially backdoor attacks, where the adversary implants backdoors to misguide the model on certain attacker-chosen sub-tasks. In this paper, we present a novel method ARIBA to accurately and robustly identify backdoor attacks in federated learning. By empirical study, we observe that backdoor attacks are discernible by the filters of CNN layers. Based on this finding, we employ unsupervised anomaly detection to evaluate the pre-processed filters and calculate an anomaly score for each client. We then identify the most suspicious clients according to their anomaly scores. Extensive experiments are conducted, which show that our method ARIBA can effectively and robustly defend against multiple state-of-the-art attacks without degrading model performance.



## **15. Adversarial Detection without Model Information**

cs.CV

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04271v1)

**Authors**: Abhishek Moitra, Youngeun Kim, Priyadarshini Panda

**Abstracts**: Most prior state-of-the-art adversarial detection works assume that the underlying vulnerable model is accessible, i,e., the model can be trained or its outputs are visible. However, this is not a practical assumption due to factors like model encryption, model information leakage and so on. In this work, we propose a model independent adversarial detection method using a simple energy function to distinguish between adversarial and natural inputs. We train a standalone detector independent of the underlying model, with sequential layer-wise training to increase the energy separation corresponding to natural and adversarial inputs. With this, we perform energy distribution-based adversarial detection. Our method achieves state-of-the-art detection performance (ROC-AUC > 0.9) across a wide range of gradient, score and decision-based adversarial attacks on CIFAR10, CIFAR100 and TinyImagenet datasets. Compared to prior approaches, our method requires ~10-100x less number of operations and parameters for adversarial detection. Further, we show that our detection method is transferable across different datasets and adversarial attacks. For reproducibility, we provide code in the supplementary material.



## **16. Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations**

cs.CV

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/2202.04235v1)

**Authors**: Yun-Yun Tsai, Lei Hsiung, Pin-Yu Chen, Tsung-Yi Ho

**Abstracts**: Model robustness against adversarial examples of single perturbation type such as the $\ell_{p}$-norm has been widely studied, yet its generalization to more realistic scenarios involving multiple semantic perturbations and their composition remains largely unexplored. In this paper, we firstly propose a novel method for generating composite adversarial examples. By utilizing component-wise projected gradient descent and automatic attack-order scheduling, our method can find the optimal attack composition. We then propose \textbf{generalized adversarial training} (\textbf{GAT}) to extend model robustness from $\ell_{p}$-norm to composite semantic perturbations, such as the combination of Hue, Saturation, Brightness, Contrast, and Rotation. The results on ImageNet and CIFAR-10 datasets show that GAT can be robust not only to any single attack but also to any combination of multiple attacks. GAT also outperforms baseline $\ell_{\infty}$-norm bounded adversarial training approaches by a significant margin.



## **17. Defeating Misclassification Attacks Against Transfer Learning**

cs.LG

This paper has been published in IEEE Transactions on Dependable and  Secure Computing.  https://doi.ieeecomputersociety.org/10.1109/TDSC.2022.3144988

**SubmitDate**: 2022-02-09    [paper-pdf](http://arxiv.org/pdf/1908.11230v4)

**Authors**: Bang Wu, Shuo Wang, Xingliang Yuan, Cong Wang, Carsten Rudolph, Xiangwen Yang

**Abstracts**: Transfer learning is prevalent as a technique to efficiently generate new models (Student models) based on the knowledge transferred from a pre-trained model (Teacher model). However, Teacher models are often publicly available for sharing and reuse, which inevitably introduces vulnerability to trigger severe attacks against transfer learning systems. In this paper, we take a first step towards mitigating one of the most advanced misclassification attacks in transfer learning. We design a distilled differentiator via activation-based network pruning to enervate the attack transferability while retaining accuracy. We adopt an ensemble structure from variant differentiators to improve the defence robustness. To avoid the bloated ensemble size during inference, we propose a two-phase defence, in which inference from the Student model is firstly performed to narrow down the candidate differentiators to be assembled, and later only a small, fixed number of them can be chosen to validate clean or reject adversarial inputs effectively. Our comprehensive evaluations on both large and small image recognition tasks confirm that the Student models with our defence of only 5 differentiators are immune to over 90% of the adversarial inputs with an accuracy loss of less than 10%. Our comparison also demonstrates that our design outperforms prior problematic defences.



## **18. Ontology-based Attack Graph Enrichment**

cs.CR

18 pages, 3 figures, 1 table, conference paper (TIEMS Annual  Conference, December 2021, Paris, France)

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.04016v1)

**Authors**: Kéren Saint-Hilaire, Frédéric Cuppens, Nora Cuppens, Joaquin Garcia-Alfaro

**Abstracts**: Attack graphs provide a representation of possible actions that adversaries can perpetrate to attack a system. They are used by cybersecurity experts to make decisions, e.g., to decide remediation and recovery plans. Different approaches can be used to build such graphs. We focus on logical attack graphs, based on predicate logic, to define the causality of adversarial actions. Since networks and vulnerabilities are constantly changing (e.g., new applications get installed on system devices, updated services get publicly exposed, etc.), we propose to enrich the attack graph generation approach with a semantic augmentation post-processing of the predicates. Graphs are now mapped to monitoring alerts confirming successful attack actions and updated according to network and vulnerability changes. As a result, predicates get periodically updated, based on attack evidences and ontology enrichment. This allows to verify whether changes lead the attacker to the initial goals or to cause further damage to the system not anticipated in the initial graphs. We illustrate the approach under the specific domain of cyber-physical security affecting smart cities. We validate the approach using existing tools and ontologies.



## **19. Verification-Aided Deep Ensemble Selection**

cs.LG

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.03898v1)

**Authors**: Guy Amir, Guy Katz, Michael Schapira

**Abstracts**: Deep neural networks (DNNs) have become the technology of choice for realizing a variety of complex tasks. However, as highlighted by many recent studies, even an imperceptible perturbation to a correctly classified input can lead to misclassification by a DNN. This renders DNNs vulnerable to strategic input manipulations by attackers, and also prone to oversensitivity to environmental noise.   To mitigate this phenomenon, practitioners apply joint classification by an ensemble of DNNs. By aggregating the classification outputs of different individual DNNs for the same input, ensemble-based classification reduces the risk of misclassifications due to the specific realization of the stochastic training process of any single DNN. However, the effectiveness of a DNN ensemble is highly dependent on its members not simultaneously erring on many different inputs.   In this case study, we harness recent advances in DNN verification to devise a methodology for identifying ensemble compositions that are less prone to simultaneous errors, even when the input is adversarially perturbed -- resulting in more robustly-accurate ensemble-based classification.   Our proposed framework uses a DNN verifier as a backend, and includes heuristics that help reduce the high complexity of directly verifying ensembles. More broadly, our work puts forth a novel universal objective for formal verification that can potentially improve the robustness of real-world, deep-learning-based systems across a variety of application domains.



## **20. Invertible Tabular GANs: Killing Two Birds with OneStone for Tabular Data Synthesis**

cs.LG

19 pages

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.03636v1)

**Authors**: Jaehoon Lee, Jihyeon Hyeong, Jinsung Jeon, Noseong Park, Jihoon Cho

**Abstracts**: Tabular data synthesis has received wide attention in the literature. This is because available data is often limited, incomplete, or cannot be obtained easily, and data privacy is becoming increasingly important. In this work, we present a generalized GAN framework for tabular synthesis, which combines the adversarial training of GANs and the negative log-density regularization of invertible neural networks. The proposed framework can be used for two distinctive objectives. First, we can further improve the synthesis quality, by decreasing the negative log-density of real records in the process of adversarial training. On the other hand, by increasing the negative log-density of real records, realistic fake records can be synthesized in a way that they are not too much close to real records and reduce the chance of potential information leakage. We conduct experiments with real-world datasets for classification, regression, and privacy attacks. In general, the proposed method demonstrates the best synthesis quality (in terms of task-oriented evaluation metrics, e.g., F1) when decreasing the negative log-density during the adversarial training. If increasing the negative log-density, our experimental results show that the distance between real and fake records increases, enhancing robustness against privacy attacks.



## **21. A Survey on Poisoning Attacks Against Supervised Machine Learning**

cs.CR

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2202.02510v2)

**Authors**: Wenjun Qiu

**Abstracts**: With the rise of artificial intelligence and machine learning in modern computing, one of the major concerns regarding such techniques is to provide privacy and security against adversaries. We present this survey paper to cover the most representative papers in poisoning attacks against supervised machine learning models. We first provide a taxonomy to categorize existing studies and then present detailed summaries for selected papers. We summarize and compare the methodology and limitations of existing literature. We conclude this paper with potential improvements and future directions to further exploit and prevent poisoning attacks on supervised models. We propose several unanswered research questions to encourage and inspire researchers for future work.



## **22. Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks**

cs.LG

Accepted at AAAI 2022. This version contains considerably extended  results in the L0 threat model

**SubmitDate**: 2022-02-08    [paper-pdf](http://arxiv.org/pdf/2006.12834v3)

**Authors**: Francesco Croce, Maksym Andriushchenko, Naman D. Singh, Nicolas Flammarion, Matthias Hein

**Abstracts**: We propose a versatile framework based on random search, Sparse-RS, for score-based sparse targeted and untargeted attacks in the black-box setting. Sparse-RS does not rely on substitute models and achieves state-of-the-art success rate and query efficiency for multiple sparse attack models: $l_0$-bounded perturbations, adversarial patches, and adversarial frames. The $l_0$-version of untargeted Sparse-RS outperforms all black-box and even all white-box attacks for different models on MNIST, CIFAR-10, and ImageNet. Moreover, our untargeted Sparse-RS achieves very high success rates even for the challenging settings of $20\times20$ adversarial patches and $2$-pixel wide adversarial frames for $224\times224$ images. Finally, we show that Sparse-RS can be applied to generate targeted universal adversarial patches where it significantly outperforms the existing approaches. The code of our framework is available at https://github.com/fra31/sparse-rs.



## **23. Evaluating Robustness of Cooperative MARL: A Model-based Approach**

cs.LG

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03558v1)

**Authors**: Nhan H. Pham, Lam M. Nguyen, Jie Chen, Hoang Thanh Lam, Subhro Das, Tsui-Wei Weng

**Abstracts**: In recent years, a proliferation of methods were developed for cooperative multi-agent reinforcement learning (c-MARL). However, the robustness of c-MARL agents against adversarial attacks has been rarely explored. In this paper, we propose to evaluate the robustness of c-MARL agents via a model-based approach. Our proposed formulation can craft stronger adversarial state perturbations of c-MARL agents(s) to lower total team rewards more than existing model-free approaches. In addition, we propose the first victim-agent selection strategy which allows us to develop even stronger adversarial attack. Numerical experiments on multi-agent MuJoCo benchmarks illustrate the advantage of our approach over other baselines. The proposed model-based attack consistently outperforms other baselines in all tested environments.



## **24. Deletion Inference, Reconstruction, and Compliance in Machine (Un)Learning**

cs.LG

Full version of a paper appearing in the 22nd Privacy Enhancing  Technologies Symposium (PETS 2022)

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03460v1)

**Authors**: Ji Gao, Sanjam Garg, Mohammad Mahmoody, Prashant Nalini Vasudevan

**Abstracts**: Privacy attacks on machine learning models aim to identify the data that is used to train such models. Such attacks, traditionally, are studied on static models that are trained once and are accessible by the adversary. Motivated to meet new legal requirements, many machine learning methods are recently extended to support machine unlearning, i.e., updating models as if certain examples are removed from their training sets, and meet new legal requirements. However, privacy attacks could potentially become more devastating in this new setting, since an attacker could now access both the original model before deletion and the new model after the deletion. In fact, the very act of deletion might make the deleted record more vulnerable to privacy attacks.   Inspired by cryptographic definitions and the differential privacy framework, we formally study privacy implications of machine unlearning. We formalize (various forms of) deletion inference and deletion reconstruction attacks, in which the adversary aims to either identify which record is deleted or to reconstruct (perhaps part of) the deleted records. We then present successful deletion inference and reconstruction attacks for a variety of machine learning models and tasks such as classification, regression, and language models. Finally, we show that our attacks would provably be precluded if the schemes satisfy (variants of) Deletion Compliance (Garg, Goldwasser, and Vasudevan, Eurocrypt' 20).



## **25. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

stat.ML

30 pages

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03397v1)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstracts**: We analyze a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. We show that without warm-start, it is still possible to achieve order-wise optimal and near-optimal sample complexity for the stochastic and deterministic settings, respectively. In particular, we propose a simple method which uses stochastic fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Compared to methods using warm-start, ours is better suited for meta-learning and yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates.



## **26. Membership Inference Attacks and Defenses in Neural Network Pruning**

cs.CR

This paper has been conditionally accepted to USENIX Security  Symposium 2022. This is an extended version

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03335v1)

**Authors**: Xiaoyong Yuan, Lan Zhang

**Abstracts**: Neural network pruning has been an essential technique to reduce the computation and memory requirements for using deep neural networks for resource-constrained devices. Most existing research focuses primarily on balancing the sparsity and accuracy of a pruned neural network by strategically removing insignificant parameters and retraining the pruned model. Such efforts on reusing training samples pose serious privacy risks due to increased memorization, which, however, has not been investigated yet.   In this paper, we conduct the first analysis of privacy risks in neural network pruning. Specifically, we investigate the impacts of neural network pruning on training data privacy, i.e., membership inference attacks. We first explore the impact of neural network pruning on prediction divergence, where the pruning process disproportionately affects the pruned model's behavior for members and non-members. Meanwhile, the influence of divergence even varies among different classes in a fine-grained manner. Enlighten by such divergence, we proposed a self-attention membership inference attack against the pruned neural networks. Extensive experiments are conducted to rigorously evaluate the privacy impacts of different pruning approaches, sparsity levels, and adversary knowledge. The proposed attack shows the higher attack performance on the pruned models when compared with eight existing membership inference attacks. In addition, we propose a new defense mechanism to protect the pruning process by mitigating the prediction divergence based on KL-divergence distance, whose effectiveness has been experimentally demonstrated to effectively mitigate the privacy risks while maintaining the sparsity and accuracy of the pruned models.



## **27. On The Empirical Effectiveness of Unrealistic Adversarial Hardening Against Realistic Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03277v1)

**Authors**: Salijona Dyrmishi, Salah Ghamizi, Thibault Simonetto, Yves Le Traon, Maxime Cordy

**Abstracts**: While the literature on security attacks and defense of Machine Learning (ML) systems mostly focuses on unrealistic adversarial examples, recent research has raised concern about the under-explored field of realistic adversarial attacks and their implications on the robustness of real-world systems. Our paper paves the way for a better understanding of adversarial robustness against realistic attacks and makes two major contributions. First, we conduct a study on three real-world use cases (text classification, botnet detection, malware detection)) and five datasets in order to evaluate whether unrealistic adversarial examples can be used to protect models against realistic examples. Our results reveal discrepancies across the use cases, where unrealistic examples can either be as effective as the realistic ones or may offer only limited improvement. Second, to explain these results, we analyze the latent representation of the adversarial examples generated with realistic and unrealistic attacks. We shed light on the patterns that discriminate which unrealistic examples can be used for effective hardening. We release our code, datasets and models to support future research in exploring how to reduce the gap between unrealistic and realistic adversarial attacks.



## **28. Strong Converse Theorem for Source Encryption under Side-Channel Attacks**

cs.IT

9 pages, 6 figures. The short version of this paper was submitted to  ISIT2022, arXiv admin note: text overlap with arXiv:1801.02563

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2201.11670v3)

**Authors**: Yasutada Oohama, Bagus Santoso

**Abstracts**: We are interested in investigating the security of source encryption with a symmetric key under side-channel attacks. In this paper, we propose a general framework of source encryption with a symmetric key under the side-channel attacks, which applies to \emph{any} source encryption with a symmetric key and \emph{any} kind of side-channel attacks targeting the secret key. We also propose a new security criterion for strong secrecy under side-channel attacks, which is a natural extension of mutual information, i.e., \emph{the maximum conditional mutual information between the plaintext and the ciphertext given the adversarial key leakage, where the maximum is taken over all possible plaintext distribution}. Under this new criterion, we successfully formulate the rate region, which serves as both necessary and sufficient conditions to have secure transmission even under side-channel attacks. Furthermore, we also prove another theoretical result on our new security criterion, which might be interesting in its own right: in the case of the discrete memoryless source, no perfect secrecy under side-channel attacks in the standard security criterion, i.e., the ordinary mutual information, is achievable without achieving perfect secrecy in this new security criterion, although our new security criterion is more strict than the standard security criterion.



## **29. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

cs.CR

18 pages, 11 figures

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03195v1)

**Authors**: Jing Xu, Rui Wang, Kaitai Liang, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although many research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). CBA is conducted by embedding the same global trigger during training for every malicious party, while DBA is conducted by decomposing a global trigger into separate local triggers and embedding them into the training dataset of different malicious parties, respectively. Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases, while rarely, the DBA attack performance is close to CBA. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for different trigger sizes, poisoning intensities, and trigger densities, with trigger density being the most influential.



## **30. Transformers in Self-Supervised Monocular Depth Estimation with Unknown Camera Intrinsics**

cs.CV

Published in 17th International Conference on Computer Vision Theory  and Applications (VISAP, 2022)

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03131v1)

**Authors**: Arnav Varma, Hemang Chawla, Bahram Zonooz, Elahe Arani

**Abstracts**: The advent of autonomous driving and advanced driver assistance systems necessitates continuous developments in computer vision for 3D scene understanding. Self-supervised monocular depth estimation, a method for pixel-wise distance estimation of objects from a single camera without the use of ground truth labels, is an important task in 3D scene understanding. However, existing methods for this task are limited to convolutional neural network (CNN) architectures. In contrast with CNNs that use localized linear operations and lose feature resolution across the layers, vision transformers process at constant resolution with a global receptive field at every stage. While recent works have compared transformers against their CNN counterparts for tasks such as image classification, no study exists that investigates the impact of using transformers for self-supervised monocular depth estimation. Here, we first demonstrate how to adapt vision transformers for self-supervised monocular depth estimation. Thereafter, we compare the transformer and CNN-based architectures for their performance on KITTI depth prediction benchmarks, as well as their robustness to natural corruptions and adversarial attacks, including when the camera intrinsics are unknown. Our study demonstrates how transformer-based architecture, though lower in run-time efficiency, achieves comparable performance while being more robust and generalizable.



## **31. Adversarial Attacks and Defense for Non-Parametric Two-Sample Tests**

cs.LG

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2202.03077v1)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstracts**: Non-parametric two-sample tests (TSTs) that judge whether two sets of samples are drawn from the same distribution, have been widely used in the analysis of critical data. People tend to employ TSTs as trusted basic tools and rarely have any doubt about their reliability. This paper systematically uncovers the failure mode of non-parametric TSTs through adversarial attacks and then proposes corresponding defense strategies. First, we theoretically show that an adversary can upper-bound the distributional shift which guarantees the attack's invisibility. Furthermore, we theoretically find that the adversary can also degrade the lower bound of a TST's test power, which enables us to iteratively minimize the test criterion in order to search for adversarial pairs. To enable TST-agnostic attacks, we propose an ensemble attack (EA) framework that jointly minimizes the different types of test criteria. Second, to robustify TSTs, we propose a max-min optimization that iteratively generates adversarial pairs to train the deep kernels. Extensive experiments on both simulated and real-world datasets validate the adversarial vulnerabilities of non-parametric TSTs and the effectiveness of our proposed defense.



## **32. Demystifying the Transferability of Adversarial Attacks in Computer Networks**

cs.CR

15 pages

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2110.04488v2)

**Authors**: Ehsan Nowroozi, Yassine Mekdad, Mohammad Hajian Berenjestanaki, Mauro Conti, Abdeslam EL Fergougui

**Abstracts**: Convolutional Neural Networks (CNNs) models are one of the most frequently used deep learning networks and are extensively used in both academia and industry. Recent studies demonstrated that adversarial attacks against such models can maintain their effectiveness even when used on models other than the one targeted by the attacker. This major property is known as transferability and makes CNNs ill-suited for security applications. In this paper, we provide the first comprehensive study which assesses the robustness of CNN-based models for computer networks against adversarial transferability. Furthermore, we investigate whether the transferability property issue holds in computer networks applications. In our experiments, we first consider five different attacks: the Iterative Fast Gradient Method (I-FGSM), the Jacobian-based Saliency Map (JSMA), the Limited-memory Broyden letcher Goldfarb Shanno BFGS (L-BFGS), the Projected Gradient Descent (PGD), and the DeepFool attack. Then, we perform these attacks against three well-known datasets: the Network-based Detection of IoT (N-BaIoT) dataset and the Domain Generating Algorithms (DGA) dataset, and RIPE Atlas dataset. Our experimental results show clearly that the transferability happens in specific use cases for the I-FGSM, the JSMA, and the LBFGS attack. In such scenarios, the attack success rate on the target network range from 63.00\% to 100\%. Finally, we suggest two shielding strategies to hinder the attack transferability, by considering the Most Powerful Attacks (MPAs), and the mismatch LSTM architecture.



## **33. Explaining Adversarial Vulnerability with a Data Sparsity Hypothesis**

cs.AI

**SubmitDate**: 2022-02-07    [paper-pdf](http://arxiv.org/pdf/2103.00778v2)

**Authors**: Mahsa Paknezhad, Cuong Phuc Ngo, Amadeus Aristo Winarto, Alistair Cheong, Chuen Yang Beh, Jiayang Wu, Hwee Kuan Lee

**Abstracts**: Despite many proposed algorithms to provide robustness to deep learning (DL) models, DL models remain susceptible to adversarial attacks. We hypothesize that the adversarial vulnerability of DL models stems from two factors. The first factor is data sparsity which is that in the high dimensional input data space, there exist large regions outside the support of the data distribution. The second factor is the existence of many redundant parameters in the DL models. Owing to these factors, different models are able to come up with different decision boundaries with comparably high prediction accuracy. The appearance of the decision boundaries in the space outside the support of the data distribution does not affect the prediction accuracy of the model. However, it makes an important difference in the adversarial robustness of the model. We hypothesize that the ideal decision boundary is as far as possible from the support of the data distribution. In this paper, we develop a training framework to observe if DL models are able to learn such a decision boundary spanning the space around the class distributions further from the data points themselves. Semi-supervised learning was deployed during training by leveraging unlabeled data generated in the space outside the support of the data distribution. We measured adversarial robustness of the models trained using this training framework against well-known adversarial attacks and by using robustness metrics. We found that models trained using our framework, as well as other regularization methods and adversarial training support our hypothesis of data sparsity and that models trained with these methods learn to have decision boundaries more similar to the aforementioned ideal decision boundary. The code for our training framework is available at https://github.com/MahsaPaknezhad/AdversariallyRobustTraining.



## **34. Adversarial Unlearning of Backdoors via Implicit Hypergradient**

cs.LG

In proceeding of the Tenth International Conference on Learning  Representations (ICLR 2022)

**SubmitDate**: 2022-02-06    [paper-pdf](http://arxiv.org/pdf/2110.03735v4)

**Authors**: Yi Zeng, Si Chen, Won Park, Z. Morley Mao, Ming Jin, Ruoxi Jia

**Abstracts**: We propose a minimax formulation for removing backdoors from a given poisoned model based on a small set of clean data. This formulation encompasses much of prior work on backdoor removal. We propose the Implicit Bacdoor Adversarial Unlearning (I-BAU) algorithm to solve the minimax. Unlike previous work, which breaks down the minimax into separate inner and outer problems, our algorithm utilizes the implicit hypergradient to account for the interdependence between inner and outer optimization. We theoretically analyze its convergence and the generalizability of the robustness gained by solving minimax on clean data to unseen test data. In our evaluation, we compare I-BAU with six state-of-art backdoor defenses on seven backdoor attacks over two datasets and various attack settings, including the common setting where the attacker targets one class as well as important but underexplored settings where multiple classes are targeted. I-BAU's performance is comparable to and most often significantly better than the best baseline. Particularly, its performance is more robust to the variation on triggers, attack settings, poison ratio, and clean data size. Moreover, I-BAU requires less computation to take effect; particularly, it is more than $13\times$ faster than the most efficient baseline in the single-target attack setting. Furthermore, it can remain effective in the extreme case where the defender can only access 100 clean samples -- a setting where all the baselines fail to produce acceptable results.



## **35. Pipe Overflow: Smashing Voice Authentication for Fun and Profit**

cs.LG

**SubmitDate**: 2022-02-06    [paper-pdf](http://arxiv.org/pdf/2202.02751v1)

**Authors**: Shimaa Ahmed, Yash Wani, Ali Shahin Shamsabadi, Mohammad Yaghini, Ilia Shumailov, Nicolas Papernot, Kassem Fawaz

**Abstracts**: Recent years have seen a surge of popularity of acoustics-enabled personal devices powered by machine learning. Yet, machine learning has proven to be vulnerable to adversarial examples. Large number of modern systems protect themselves against such attacks by targeting the artificiality, i.e., they deploy mechanisms to detect the lack of human involvement in generating the adversarial examples. However, these defenses implicitly assume that humans are incapable of producing meaningful and targeted adversarial examples. In this paper, we show that this base assumption is wrong. In particular, we demonstrate that for tasks like speaker identification, a human is capable of producing analog adversarial examples directly with little cost and supervision: by simply speaking through a tube, an adversary reliably impersonates other speakers in eyes of ML models for speaker identification. Our findings extend to a range of other acoustic-biometric tasks such as liveness, bringing into question their use in security-critical settings in real life, such as phone banking.



## **36. EvadeDroid: A Practical Evasion Attack on Machine Learning for Black-box Android Malware Detection**

cs.LG

**SubmitDate**: 2022-02-05    [paper-pdf](http://arxiv.org/pdf/2110.03301v2)

**Authors**: Hamid Bostani, Veelasha Moonsamy

**Abstracts**: Over the last decade, several studies have investigated the weaknesses of Android malware detectors against adversarial examples by proposing novel evasion attacks; however, their practicality in manipulating real-world malware remains arguable. The majority of studies have assumed attackers know the details of the target classifiers used for malware detection, while in reality, malicious actors have limited access to the target classifiers. This paper presents a practical evasion attack, EvadeDroid, to circumvent black-box Android malware detectors. In addition to generating real-world adversarial malware, the proposed evasion attack can also preserve the functionality of the original malware samples. EvadeDroid prepares a collection of functionality-preserving transformations using an n-gram-based similarity method, which are then used to morph malware instances into benign ones via an iterative and incremental manipulation strategy. The proposed manipulation technique is a novel, query-efficient optimization algorithm with the aim of finding and injecting optimal sequences of transformations into malware samples. Our empirical evaluation demonstrates the efficacy of EvadeDroid under hard- and soft-label attacks. Moreover, EvadeDroid is capable to generate practical adversarial examples with only a small number of queries, with evasion rates of $81\%$, $73\%$, $75\%$, and $79\%$ for DREBIN, Sec-SVM, MaMaDroid, and ADE-MA, respectively. Finally, we show that EvadeDroid is able to preserve its stealthiness against five popular commercial antivirus, thus demonstrating its feasibility in the real world.



## **37. Iota: A Framework for Analyzing System-Level Security of IoTs**

cs.CR

This manuscript has been accepted by IoTDI 2022

**SubmitDate**: 2022-02-05    [paper-pdf](http://arxiv.org/pdf/2202.02506v1)

**Authors**: Zheng Fang, Hao Fu, Tianbo Gu, Pengfei Hu, Jinyue Song, Trent Jaeger, Prasant Mohapatra

**Abstracts**: Most IoT systems involve IoT devices, communication protocols, remote cloud, IoT applications, mobile apps, and the physical environment. However, existing IoT security analyses only focus on a subset of all the essential components, such as device firmware, and ignore IoT systems' interactive nature, resulting in limited attack detection capabilities. In this work, we propose Iota, a logic programming-based framework to perform system-level security analysis for IoT systems. Iota generates attack graphs for IoT systems, showing all of the system resources that can be compromised and enumerating potential attack traces. In building Iota, we design novel techniques to scan IoT systems for individual vulnerabilities and further create generic exploit models for IoT vulnerabilities. We also identify and model physical dependencies between different devices as they are unique to IoT systems and are employed by adversaries to launch complicated attacks. In addition, we utilize NLP techniques to extract IoT app semantics based on app descriptions. To evaluate vulnerabilities' system-wide impact, we propose two metrics based on the attack graph, which provide guidance on fortifying IoT systems. Evaluation on 127 IoT CVEs (Common Vulnerabilities and Exposures) shows that Iota's exploit modeling module achieves over 80% accuracy in predicting vulnerabilities' preconditions and effects. We apply Iota to 37 synthetic smart home IoT systems based on real-world IoT apps and devices. Experimental results show that our framework is effective and highly efficient. Among 27 shortest attack traces revealed by the attack graphs, 62.8% are not anticipated by the system administrator. It only takes 1.2 seconds to generate and analyze the attack graph for an IoT system consisting of 50 devices.



## **38. Improving Ensemble Robustness by Collaboratively Promoting and Demoting Adversarial Robustness**

cs.CV

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2009.09612v2)

**Authors**: Anh Bui, Trung Le, He Zhao, Paul Montague, Olivier deVel, Tamas Abraham, Dinh Phung

**Abstracts**: Ensemble-based adversarial training is a principled approach to achieve robustness against adversarial attacks. An important technique of this approach is to control the transferability of adversarial examples among ensemble members. We propose in this work a simple yet effective strategy to collaborate among committee models of an ensemble model. This is achieved via the secure and insecure sets defined for each model member on a given sample, hence help us to quantify and regularize the transferability. Consequently, our proposed framework provides the flexibility to reduce the adversarial transferability as well as to promote the diversity of ensemble members, which are two crucial factors for better robustness in our ensemble approach. We conduct extensive and comprehensive experiments to demonstrate that our proposed method outperforms the state-of-the-art ensemble baselines, at the same time can detect a wide range of adversarial examples with a nearly perfect accuracy. Our code is available at: https://github.com/tuananhbui89/Crossing-Collaborative-Ensemble.



## **39. Temporal Motifs in Patent Opposition and Collaboration Networks**

cs.SI

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2110.11198v2)

**Authors**: Penghang Liu, Naoki Masuda, Tomomi Kito, A. Erdem Sarıyüce

**Abstracts**: Patents are intellectual properties that reflect innovative activities of companies and organizations. The literature is rich with the studies that analyze the citations among the patents and the collaboration relations among companies that own the patents. However, the adversarial relations between the patent owners are not as well investigated. One proxy to model such relations is the patent opposition, which is a legal activity in which a company challenges the validity of a patent. Characterizing the patent oppositions, collaborations, and the interplay between them can help better understand the companies' business strategies. Temporality matters in this context as the order and frequency of oppositions and collaborations characterize their interplay. In this study, we construct a two-layer temporal network to model the patent oppositions and collaborations among the companies. We utilize temporal motifs to analyze the oppositions and collaborations from structural and temporal perspectives. We first characterize the frequent motifs in patent oppositions and investigate how often the companies of different sizes attack other companies. We show that large companies tend to engage in opposition with multiple companies. Then we analyze the temporal interplay between collaborations and oppositions. We find that two adversarial companies are more likely to collaborate in the future than two collaborating companies oppose each other in the future.



## **40. Dikaios: Privacy Auditing of Algorithmic Fairness via Attribute Inference Attacks**

cs.CR

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2202.02242v1)

**Authors**: Jan Aalmoes, Vasisht Duddu, Antoine Boutet

**Abstracts**: Machine learning (ML) models have been deployed for high-stakes applications. Due to class imbalance in the sensitive attribute observed in the datasets, ML models are unfair on minority subgroups identified by a sensitive attribute, such as race and sex. In-processing fairness algorithms ensure model predictions are independent of sensitive attribute. Furthermore, ML models are vulnerable to attribute inference attacks where an adversary can identify the values of sensitive attribute by exploiting their distinguishable model predictions. Despite privacy and fairness being important pillars of trustworthy ML, the privacy risk introduced by fairness algorithms with respect to attribute leakage has not been studied. We identify attribute inference attacks as an effective measure for auditing blackbox fairness algorithms to enable model builder to account for privacy and fairness in the model design. We proposed Dikaios, a privacy auditing tool for fairness algorithms for model builders which leveraged a new effective attribute inference attack that account for the class imbalance in sensitive attributes through an adaptive prediction threshold. We evaluated Dikaios to perform a privacy audit of two in-processing fairness algorithms over five datasets. We show that our attribute inference attacks with adaptive prediction threshold significantly outperform prior attacks. We highlighted the limitations of in-processing fairness algorithms to ensure indistinguishable predictions across different values of sensitive attributes. Indeed, the attribute privacy risk of these in-processing fairness schemes is highly variable according to the proportion of the sensitive attributes in the dataset. This unpredictable effect of fairness mechanisms on the attribute privacy risk is an important limitation on their utilization which has to be accounted by the model builder.



## **41. Sparse Polynomial Optimisation for Neural Network Verification**

eess.SY

25 pages, 20 figures

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2202.02241v1)

**Authors**: Matthew Newton, Antonis Papachristodoulou

**Abstracts**: The prevalence of neural networks in society is expanding at an increasing rate. It is becoming clear that providing robust guarantees on systems that use neural networks is very important, especially in safety-critical applications. A trained neural network's sensitivity to adversarial attacks is one of its greatest shortcomings. To provide robust guarantees, one popular method that has seen success is to bound the activation functions using equality and inequality constraints. However, there are numerous ways to form these bounds, providing a trade-off between conservativeness and complexity. Depending on the complexity of these bounds, the computational time of the optimisation problem varies, with longer solve times often leading to tighter bounds. We approach the problem from a different perspective, using sparse polynomial optimisation theory and the Positivstellensatz, which derives from the field of real algebraic geometry. The former exploits the natural cascading structure of the neural network using ideas from chordal sparsity while the later asserts the emptiness of a semi-algebraic set with a nested family of tests of non-decreasing accuracy to provide tight bounds. We show that bounds can be tightened significantly, whilst the computational time remains reasonable. We compare the solve times of different solvers and show how the accuracy can be improved at the expense of increased computation time. We show that by using this sparse polynomial framework the solve time and accuracy can be improved over other methods for neural network verification with ReLU, sigmoid and tanh activation functions.



## **42. Pixle: a fast and effective black-box attack based on rearranging pixels**

cs.LG

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2202.02236v1)

**Authors**: Jary Pomponi, Simone Scardapane, Aurelio Uncini

**Abstracts**: Recent research has found that neural networks are vulnerable to several types of adversarial attacks, where the input samples are modified in such a way that the model produces a wrong prediction that misclassifies the adversarial sample. In this paper we focus on black-box adversarial attacks, that can be performed without knowing the inner structure of the attacked model, nor the training procedure, and we propose a novel attack that is capable of correctly attacking a high percentage of samples by rearranging a small number of pixels within the attacked image. We demonstrate that our attack works on a large number of datasets and models, that it requires a small number of iterations, and that the distance between the original sample and the adversarial one is negligible to the human eye.



## **43. Modeling Adversarial Noise for Adversarial Training**

cs.LG

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2109.09901v4)

**Authors**: Dawei Zhou, Nannan Wang, Bo Han, Tongliang Liu

**Abstracts**: Deep neural networks have been demonstrated to be vulnerable to adversarial noise, promoting the development of defense against adversarial attacks. Motivated by the fact that adversarial noise contains well-generalizing features and that the relationship between adversarial data and natural data can help infer natural data and make reliable predictions, in this paper, we study to model adversarial noise by learning the transition relationship between adversarial labels (i.e. the flipped labels used to generate adversarial data) and natural labels (i.e. the ground truth labels of the natural data). Specifically, we introduce an instance-dependent transition matrix to relate adversarial labels and natural labels, which can be seamlessly embedded with the target model (enabling us to model stronger adaptive adversarial noise). Empirical evaluations demonstrate that our method could effectively improve adversarial accuracy.



## **44. Knowledge Cross-Distillation for Membership Privacy**

cs.CR

Accepted by PETS 2022

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2111.01363v3)

**Authors**: Rishav Chourasia, Batnyam Enkhtaivan, Kunihiro Ito, Junki Mori, Isamu Teranishi, Hikaru Tsuchida

**Abstracts**: A membership inference attack (MIA) poses privacy risks for the training data of a machine learning model. With an MIA, an attacker guesses if the target data are a member of the training dataset. The state-of-the-art defense against MIAs, distillation for membership privacy (DMP), requires not only private data for protection but a large amount of unlabeled public data. However, in certain privacy-sensitive domains, such as medicine and finance, the availability of public data is not guaranteed. Moreover, a trivial method for generating public data by using generative adversarial networks significantly decreases the model accuracy, as reported by the authors of DMP. To overcome this problem, we propose a novel defense against MIAs that uses knowledge distillation without requiring public data. Our experiments show that the privacy protection and accuracy of our defense are comparable to those of DMP for the benchmark tabular datasets used in MIA research, Purchase100 and Texas100, and our defense has a much better privacy-utility trade-off than those of the existing defenses that also do not use public data for the image dataset CIFAR10.



## **45. Fast Gradient Non-sign Methods**

cs.CV

**SubmitDate**: 2022-02-04    [paper-pdf](http://arxiv.org/pdf/2110.12734v3)

**Authors**: Yaya Cheng, Jingkuan Song, Xiaosu Zhu, Qilong Zhang, Lianli Gao, Heng Tao Shen

**Abstracts**: Adversarial attacks make their success in DNNs, and among them, gradient-based algorithms become one of the mainstreams. Based on the linearity hypothesis, under $\ell_\infty$ constraint, $sign$ operation applied to the gradients is a good choice for generating perturbations. However, side-effects from such operation exist since it leads to the bias of direction between real gradients and perturbations. In other words, current methods contain a gap between real gradients and actual noises, which leads to biased and inefficient attacks. Therefore in this paper, based on the Taylor expansion, the bias is analyzed theoretically, and the correction of $sign$, i.e., Fast Gradient Non-sign Method (FGNM), is further proposed. Notably, FGNM is a general routine that seamlessly replaces the conventional $sign$ operation in gradient-based attacks with negligible extra computational cost. Extensive experiments demonstrate the effectiveness of our methods. Specifically, for untargeted black-box attacks, ours outperform them by 27.5% at most and 9.5% on average. For targeted attacks against defense models, it is 15.1% and 12.7%. Our anonymous code is publicly available at https://github.com/yaya-cheng/FGNM



## **46. FRL: Federated Rank Learning**

cs.LG

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2110.04350v2)

**Authors**: Hamid Mozaffari, Virat Shejwalkar, Amir Houmansadr

**Abstracts**: Federated learning (FL) allows mutually untrusted clients to collaboratively train a common machine learning model without sharing their private/proprietary training data among each other. FL is unfortunately susceptible to poisoning by malicious clients who aim to hamper the accuracy of the commonly trained model through sending malicious model updates during FL's training process.   We argue that the key factor to the success of poisoning attacks against existing FL systems is the large space of model updates available to the clients, allowing malicious clients to search for the most poisonous model updates, e.g., by solving an optimization problem. To address this, we propose Federated Rank Learning (FRL). FRL reduces the space of client updates from model parameter updates (a continuous space of float numbers) in standard FL to the space of parameter rankings (a discrete space of integer values). To be able to train the global model using parameter ranks (instead of parameter weights), FRL leverage ideas from recent supermasks training mechanisms. Specifically, FRL clients rank the parameters of a randomly initialized neural network (provided by the server) based on their local training data. The FRL server uses a voting mechanism to aggregate the parameter rankings submitted by clients in each training epoch to generate the global ranking of the next training epoch.   Intuitively, our voting-based aggregation mechanism prevents poisoning clients from making significant adversarial modifications to the global model, as each client will have a single vote! We demonstrate the robustness of FRL to poisoning through analytical proofs and experimentation. We also show FRL's high communication efficiency. Our experiments demonstrate the superiority of FRL in real-world FL settings.



## **47. A Robust Phased Elimination Algorithm for Corruption-Tolerant Gaussian Process Bandits**

stat.ML

Preprint

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2202.01850v1)

**Authors**: Ilija Bogunovic, Zihan Li, Andreas Krause, Jonathan Scarlett

**Abstracts**: We consider the sequential optimization of an unknown, continuous, and expensive to evaluate reward function, from noisy and adversarially corrupted observed rewards. When the corruption attacks are subject to a suitable budget $C$ and the function lives in a Reproducing Kernel Hilbert Space (RKHS), the problem can be posed as corrupted Gaussian process (GP) bandit optimization. We propose a novel robust elimination-type algorithm that runs in epochs, combines exploration with infrequent switching to select a small subset of actions, and plays each action for multiple time instants. Our algorithm, Robust GP Phased Elimination (RGP-PE), successfully balances robustness to corruptions with exploration and exploitation such that its performance degrades minimally in the presence (or absence) of adversarial corruptions. When $T$ is the number of samples and $\gamma_T$ is the maximal information gain, the corruption-dependent term in our regret bound is $O(C \gamma_T^{3/2})$, which is significantly tighter than the existing $O(C \sqrt{T \gamma_T})$ for several commonly-considered kernels. We perform the first empirical study of robustness in the corrupted GP bandit setting, and show that our algorithm is robust against a variety of adversarial attacks.



## **48. ObjectSeeker: Certifiably Robust Object Detection against Patch Hiding Attacks via Patch-agnostic Masking**

cs.CV

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2202.01811v1)

**Authors**: Chong Xiang, Alexander Valtchanov, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: Object detectors, which are widely deployed in security-critical systems such as autonomous vehicles, have been found vulnerable to physical-world patch hiding attacks. The attacker can use a single physically-realizable adversarial patch to make the object detector miss the detection of victim objects and completely undermines the functionality of object detection applications. In this paper, we propose ObjectSeeker as a defense framework for building certifiably robust object detectors against patch hiding attacks. The core operation of ObjectSeeker is patch-agnostic masking: we aim to mask out the entire adversarial patch without any prior knowledge of the shape, size, and location of the patch. This masking operation neutralizes the adversarial effect and allows any vanilla object detector to safely detect objects on the masked images. Remarkably, we develop a certification procedure to determine if ObjectSeeker can detect certain objects with a provable guarantee against any adaptive attacker within the threat model. Our evaluation with two object detectors and three datasets demonstrates a significant (~10%-40% absolute and ~2-6x relative) improvement in certified robustness over the prior work, as well as high clean performance (~1% performance drop compared with vanilla undefended models).



## **49. Toward Realistic Backdoor Injection Attacks on DNNs using Rowhammer**

cs.LG

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2110.07683v2)

**Authors**: M. Caner Tol, Saad Islam, Berk Sunar, Ziming Zhang

**Abstracts**: State-of-the-art deep neural networks (DNNs) have been proven to be vulnerable to adversarial manipulation and backdoor attacks. Backdoored models deviate from expected behavior on inputs with predefined triggers while retaining performance on clean data. Recent works focus on software simulation of backdoor injection during the inference phase by modifying network weights, which we find often unrealistic in practice due to restrictions in hardware.   In contrast, in this work for the first time we present an end-to-end backdoor injection attack realized on actual hardware on a classifier model using Rowhammer as the fault injection method. To this end, we first investigate the viability of backdoor injection attacks in real-life deployments of DNNs on hardware and address such practical issues in hardware implementation from a novel optimization perspective. We are motivated by the fact that the vulnerable memory locations are very rare, device-specific, and sparsely distributed. Consequently, we propose a novel network training algorithm based on constrained optimization to achieve a realistic backdoor injection attack in hardware. By modifying parameters uniformly across the convolutional and fully-connected layers as well as optimizing the trigger pattern together, we achieve the state-of-the-art attack performance with fewer bit flips. For instance, our method on a hardware-deployed ResNet-20 model trained on CIFAR-10 achieves over 91% test accuracy and 94% attack success rate by flipping only 10 out of 2.2 million bits.



## **50. Learnability Lock: Authorized Learnability Control Through Adversarial Invertible Transformations**

cs.LG

Accepted at ICLR 2022

**SubmitDate**: 2022-02-03    [paper-pdf](http://arxiv.org/pdf/2202.03576v1)

**Authors**: Weiqi Peng, Jinghui Chen

**Abstracts**: Owing much to the revolution of information technology, the recent progress of deep learning benefits incredibly from the vastly enhanced access to data available in various digital formats. However, in certain scenarios, people may not want their data being used for training commercial models and thus studied how to attack the learnability of deep learning models. Previous works on learnability attack only consider the goal of preventing unauthorized exploitation on the specific dataset but not the process of restoring the learnability for authorized cases. To tackle this issue, this paper introduces and investigates a new concept called "learnability lock" for controlling the model's learnability on a specific dataset with a special key. In particular, we propose adversarial invertible transformation, that can be viewed as a mapping from image to image, to slightly modify data samples so that they become "unlearnable" by machine learning models with negligible loss of visual features. Meanwhile, one can unlock the learnability of the dataset and train models normally using the corresponding key. The proposed learnability lock leverages class-wise perturbation that applies a universal transformation function on data samples of the same label. This ensures that the learnability can be easily restored with a simple inverse transformation while remaining difficult to be detected or reverse-engineered. We empirically demonstrate the success and practicability of our method on visual classification tasks.



