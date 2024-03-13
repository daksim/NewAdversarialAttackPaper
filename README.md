# Latest Adversarial Attack Papers
**update at 2024-03-13 10:01:26**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Robustifying Point Cloud Networks by Refocusing**

cs.CV

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2308.05525v3) [paper-pdf](http://arxiv.org/pdf/2308.05525v3)

**Authors**: Meir Yossef Levi, Guy Gilboa

**Abstract**: The ability to cope with out-of-distribution (OOD) corruptions and adversarial attacks is crucial in real-world safety-demanding applications. In this study, we develop a general mechanism to increase neural network robustness based on focus analysis.   Recent studies have revealed the phenomenon of \textit{Overfocusing}, which leads to a performance drop. When the network is primarily influenced by small input regions, it becomes less robust and prone to misclassify under noise and corruptions.   However, quantifying overfocusing is still vague and lacks clear definitions. Here, we provide a mathematical definition of \textbf{focus}, \textbf{overfocusing} and \textbf{underfocusing}. The notions are general, but in this study, we specifically investigate the case of 3D point clouds.   We observe that corrupted sets result in a biased focus distribution compared to the clean training set.   We show that as focus distribution deviates from the one learned in the training phase - classification performance deteriorates.   We thus propose a parameter-free \textbf{refocusing} algorithm that aims to unify all corruptions under the same distribution.   We validate our findings on a 3D zero-shot classification task, achieving SOTA in robust 3D classification on ModelNet-C dataset, and in adversarial defense against Shape-Invariant attack. Code is available in: https://github.com/yossilevii100/refocusing.



## **2. Analyzing Adversarial Attacks on Sequence-to-Sequence Relevance Models**

cs.IR

13 pages, 3 figures, Accepted at ECIR 2024 as a Full Paper

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.07654v1) [paper-pdf](http://arxiv.org/pdf/2403.07654v1)

**Authors**: Andrew Parry, Maik Fröbe, Sean MacAvaney, Martin Potthast, Matthias Hagen

**Abstract**: Modern sequence-to-sequence relevance models like monoT5 can effectively capture complex textual interactions between queries and documents through cross-encoding. However, the use of natural language tokens in prompts, such as Query, Document, and Relevant for monoT5, opens an attack vector for malicious documents to manipulate their relevance score through prompt injection, e.g., by adding target words such as true. Since such possibilities have not yet been considered in retrieval evaluation, we analyze the impact of query-independent prompt injection via manually constructed templates and LLM-based rewriting of documents on several existing relevance models. Our experiments on the TREC Deep Learning track show that adversarial documents can easily manipulate different sequence-to-sequence relevance models, while BM25 (as a typical lexical model) is not affected. Remarkably, the attacks also affect encoder-only relevance models (which do not rely on natural language prompt tokens), albeit to a lesser extent.



## **3. Visual Privacy Auditing with Diffusion Models**

cs.LG

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.07588v1) [paper-pdf](http://arxiv.org/pdf/2403.07588v1)

**Authors**: Kristian Schwethelm, Johannes Kaiser, Moritz Knolle, Daniel Rueckert, Georgios Kaissis, Alexander Ziller

**Abstract**: Image reconstruction attacks on machine learning models pose a significant risk to privacy by potentially leaking sensitive information. Although defending against such attacks using differential privacy (DP) has proven effective, determining appropriate DP parameters remains challenging. Current formal guarantees on data reconstruction success suffer from overly theoretical assumptions regarding adversary knowledge about the target data, particularly in the image domain. In this work, we empirically investigate this discrepancy and find that the practicality of these assumptions strongly depends on the domain shift between the data prior and the reconstruction target. We propose a reconstruction attack based on diffusion models (DMs) that assumes adversary access to real-world image priors and assess its implications on privacy leakage under DP-SGD. We show that (1) real-world data priors significantly influence reconstruction success, (2) current reconstruction bounds do not model the risk posed by data priors well, and (3) DMs can serve as effective auditing tools for visualizing privacy leakage.



## **4. Improving deep learning with prior knowledge and cognitive models: A survey on enhancing explainability, adversarial robustness and zero-shot learning**

cs.LG

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.07078v1) [paper-pdf](http://arxiv.org/pdf/2403.07078v1)

**Authors**: Fuseinin Mumuni, Alhassan Mumuni

**Abstract**: We review current and emerging knowledge-informed and brain-inspired cognitive systems for realizing adversarial defenses, eXplainable Artificial Intelligence (XAI), and zero-shot or few-short learning. Data-driven deep learning models have achieved remarkable performance and demonstrated capabilities surpassing human experts in many applications. Yet, their inability to exploit domain knowledge leads to serious performance limitations in practical applications. In particular, deep learning systems are exposed to adversarial attacks, which can trick them into making glaringly incorrect decisions. Moreover, complex data-driven models typically lack interpretability or explainability, i.e., their decisions cannot be understood by human subjects. Furthermore, models are usually trained on standard datasets with a closed-world assumption. Hence, they struggle to generalize to unseen cases during inference in practical open-world environments, thus, raising the zero- or few-shot generalization problem. Although many conventional solutions exist, explicit domain knowledge, brain-inspired neural network and cognitive architectures offer powerful new dimensions towards alleviating these problems. Prior knowledge is represented in appropriate forms and incorporated in deep learning frameworks to improve performance. Brain-inspired cognition methods use computational models that mimic the human mind to enhance intelligent behavior in artificial agents and autonomous robots. Ultimately, these models achieve better explainability, higher adversarial robustness and data-efficient learning, and can, in turn, provide insights for cognitive science and neuroscience-that is, to deepen human understanding on how the brain works in general, and how it handles these problems.



## **5. Enhancing Adversarial Training with Prior Knowledge Distillation for Robust Image Compression**

eess.IV

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06700v1) [paper-pdf](http://arxiv.org/pdf/2403.06700v1)

**Authors**: Cao Zhi, Bao Youneng, Meng Fanyang, Li Chao, Tan Wen, Wang Genhong, Liang Yongsheng

**Abstract**: Deep neural network-based image compression (NIC) has achieved excellent performance, but NIC method models have been shown to be susceptible to backdoor attacks. Adversarial training has been validated in image compression models as a common method to enhance model robustness. However, the improvement effect of adversarial training on model robustness is limited. In this paper, we propose a prior knowledge-guided adversarial training framework for image compression models. Specifically, first, we propose a gradient regularization constraint for training robust teacher models. Subsequently, we design a knowledge distillation based strategy to generate a priori knowledge from the teacher model to the student model for guiding adversarial training. Experimental results show that our method improves the reconstruction quality by about 9dB when the Kodak dataset is elected as the backdoor attack object for psnr attack. Compared with Ma2023, our method has a 5dB higher PSNR output at high bitrate points.



## **6. PCLD: Point Cloud Layerwise Diffusion for Adversarial Purification**

cs.CV

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06698v1) [paper-pdf](http://arxiv.org/pdf/2403.06698v1)

**Authors**: Mert Gulsen, Batuhan Cengiz, Yusuf H. Sahin, Gozde Unal

**Abstract**: Point clouds are extensively employed in a variety of real-world applications such as robotics, autonomous driving and augmented reality. Despite the recent success of point cloud neural networks, especially for safety-critical tasks, it is essential to also ensure the robustness of the model. A typical way to assess a model's robustness is through adversarial attacks, where test-time examples are generated based on gradients to deceive the model. While many different defense mechanisms are studied in 2D, studies on 3D point clouds have been relatively limited in the academic field. Inspired from PointDP, which denoises the network inputs by diffusion, we propose Point Cloud Layerwise Diffusion (PCLD), a layerwise diffusion based 3D point cloud defense strategy. Unlike PointDP, we propagated the diffusion denoising after each layer to incrementally enhance the results. We apply our defense method to different types of commonly used point cloud models and adversarial attacks to evaluate its robustness. Our experiments demonstrate that the proposed defense method achieved results that are comparable to or surpass those of existing methodologies, establishing robustness through a novel technique. Code is available at https://github.com/batuceng/diffusion-layer-robustness-pc.



## **7. PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor**

cs.LG

Accepted to CVPR 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06668v1) [paper-pdf](http://arxiv.org/pdf/2403.06668v1)

**Authors**: Jaewon Jung, Hongsun Jang, Jaeyong Song, Jinho Lee

**Abstract**: Adversarial robustness of the neural network is a significant concern when it is applied to security-critical domains. In this situation, adversarial distillation is a promising option which aims to distill the robustness of the teacher network to improve the robustness of a small student network. Previous works pretrain the teacher network to make it robust to the adversarial examples aimed at itself. However, the adversarial examples are dependent on the parameters of the target network. The fixed teacher network inevitably degrades its robustness against the unseen transferred adversarial examples which targets the parameters of the student network in the adversarial distillation process. We propose PeerAiD to make a peer network learn the adversarial examples of the student network instead of adversarial examples aimed at itself. PeerAiD is an adversarial distillation that trains the peer network and the student network simultaneously in order to make the peer network specialized for defending the student network. We observe that such peer networks surpass the robustness of pretrained robust teacher network against student-attacked adversarial samples. With this peer network and adversarial distillation, PeerAiD achieves significantly higher robustness of the student network with AutoAttack (AA) accuracy up to 1.66%p and improves the natural accuracy of the student network up to 4.72%p with ResNet-18 and TinyImageNet dataset.



## **8. epsilon-Mesh Attack: A Surface-based Adversarial Point Cloud Attack for Facial Expression Recognition**

cs.CV

Accepted at 18th IEEE International Conference on Automatic Face &  Gesture Recognition (FG 2024)

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06661v1) [paper-pdf](http://arxiv.org/pdf/2403.06661v1)

**Authors**: Batuhan Cengiz, Mert Gulsen, Yusuf H. Sahin, Gozde Unal

**Abstract**: Point clouds and meshes are widely used 3D data structures for many computer vision applications. While the meshes represent the surfaces of an object, point cloud represents sampled points from the surface which is also the output of modern sensors such as LiDAR and RGB-D cameras. Due to the wide application area of point clouds and the recent advancements in deep neural networks, studies focusing on robust classification of the 3D point cloud data emerged. To evaluate the robustness of deep classifier networks, a common method is to use adversarial attacks where the gradient direction is followed to change the input slightly. The previous studies on adversarial attacks are generally evaluated on point clouds of daily objects. However, considering 3D faces, these adversarial attacks tend to affect the person's facial structure more than the desired amount and cause malformation. Specifically for facial expressions, even a small adversarial attack can have a significant effect on the face structure. In this paper, we suggest an adversarial attack called $\epsilon$-Mesh Attack, which operates on point cloud data via limiting perturbations to be on the mesh surface. We also parameterize our attack by $\epsilon$ to scale the perturbation mesh. Our surface-based attack has tighter perturbation bounds compared to $L_2$ and $L_\infty$ norm bounded attacks that operate on unit-ball. Even though our method has additional constraints, our experiments on CoMA, Bosphorus and FaceWarehouse datasets show that $\epsilon$-Mesh Attack (Perpendicular) successfully confuses trained DGCNN and PointNet models $99.72\%$ and $97.06\%$ of the time, with indistinguishable facial deformations. The code is available at https://github.com/batuceng/e-mesh-attack.



## **9. Real is not True: Backdoor Attacks Against Deepfake Detection**

cs.CR

BigDIA 2023

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06610v1) [paper-pdf](http://arxiv.org/pdf/2403.06610v1)

**Authors**: Hong Sun, Ziqiang Li, Lei Liu, Bin Li

**Abstract**: The proliferation of malicious deepfake applications has ignited substantial public apprehension, casting a shadow of doubt upon the integrity of digital media. Despite the development of proficient deepfake detection mechanisms, they persistently demonstrate pronounced vulnerability to an array of attacks. It is noteworthy that the pre-existing repertoire of attacks predominantly comprises adversarial example attack, predominantly manifesting during the testing phase. In the present study, we introduce a pioneering paradigm denominated as Bad-Deepfake, which represents a novel foray into the realm of backdoor attacks levied against deepfake detectors. Our approach hinges upon the strategic manipulation of a delimited subset of the training data, enabling us to wield disproportionate influence over the operational characteristics of a trained model. This manipulation leverages inherent frailties inherent to deepfake detectors, affording us the capacity to engineer triggers and judiciously select the most efficacious samples for the construction of the poisoned set. Through the synergistic amalgamation of these sophisticated techniques, we achieve an remarkable performance-a 100% attack success rate (ASR) against extensively employed deepfake detectors.



## **10. DNNShield: Embedding Identifiers for Deep Neural Network Ownership Verification**

cs.CR

18 pages, 11 figures, 6 tables

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06581v1) [paper-pdf](http://arxiv.org/pdf/2403.06581v1)

**Authors**: Jasper Stang, Torsten Krauß, Alexandra Dmitrienko

**Abstract**: The surge in popularity of machine learning (ML) has driven significant investments in training Deep Neural Networks (DNNs). However, these models that require resource-intensive training are vulnerable to theft and unauthorized use. This paper addresses this challenge by introducing DNNShield, a novel approach for DNN protection that integrates seamlessly before training. DNNShield embeds unique identifiers within the model architecture using specialized protection layers. These layers enable secure training and deployment while offering high resilience against various attacks, including fine-tuning, pruning, and adaptive adversarial attacks. Notably, our approach achieves this security with minimal performance and computational overhead (less than 5\% runtime increase). We validate the effectiveness and efficiency of DNNShield through extensive evaluations across three datasets and four model architectures. This practical solution empowers developers to protect their DNNs and intellectual property rights.



## **11. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2305.17000v3) [paper-pdf](http://arxiv.org/pdf/2305.17000v3)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **12. Fooling Neural Networks for Motion Forecasting via Adversarial Attacks**

cs.CV

11 pages, 8 figures, VISSAP 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.04954v2) [paper-pdf](http://arxiv.org/pdf/2403.04954v2)

**Authors**: Edgar Medina, Leyong Loh

**Abstract**: Human motion prediction is still an open problem, which is extremely important for autonomous driving and safety applications. Although there are great advances in this area, the widely studied topic of adversarial attacks has not been applied to multi-regression models such as GCNs and MLP-based architectures in human motion prediction. This work intends to reduce this gap using extensive quantitative and qualitative experiments in state-of-the-art architectures similar to the initial stages of adversarial attacks in image classification. The results suggest that models are susceptible to attacks even on low levels of perturbation. We also show experiments with 3D transformations that affect the model performance, in particular, we show that most models are sensitive to simple rotations and translations which do not alter joint distances. We conclude that similar to earlier CNN models, motion forecasting tasks are susceptible to small perturbations and simple 3D transformations.



## **13. Intra-Section Code Cave Injection for Adversarial Evasion Attacks on Windows PE Malware File**

cs.CR

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06428v1) [paper-pdf](http://arxiv.org/pdf/2403.06428v1)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam, Moustafa Saleh

**Abstract**: Windows malware is predominantly available in cyberspace and is a prime target for deliberate adversarial evasion attacks. Although researchers have investigated the adversarial malware attack problem, a multitude of important questions remain unanswered, including (a) Are the existing techniques to inject adversarial perturbations in Windows Portable Executable (PE) malware files effective enough for evasion purposes?; (b) Does the attack process preserve the original behavior of malware?; (c) Are there unexplored approaches/locations that can be used to carry out adversarial evasion attacks on Windows PE malware?; and (d) What are the optimal locations and sizes of adversarial perturbations required to evade an ML-based malware detector without significant structural change in the PE file? To answer some of these questions, this work proposes a novel approach that injects a code cave within the section (i.e., intra-section) of Windows PE malware files to make space for adversarial perturbations. In addition, a code loader is also injected inside the PE file, which reverts adversarial malware to its original form during the execution, preserving the malware's functionality and executability. To understand the effectiveness of our approach, we injected adversarial perturbations inside the .text, .data and .rdata sections, generated using the gradient descent and Fast Gradient Sign Method (FGSM), to target the two popular CNN-based malware detectors, MalConv and MalConv2. Our experiments yielded notable results, achieving a 92.31% evasion rate with gradient descent and 96.26% with FGSM against MalConv, compared to the 16.17% evasion rate for append attacks. Similarly, when targeting MalConv2, our approach achieved a remarkable maximum evasion rate of 97.93% with gradient descent and 94.34% with FGSM, significantly surpassing the 4.01% evasion rate observed with append attacks.



## **14. A Zero Trust Framework for Realization and Defense Against Generative AI Attacks in Power Grid**

cs.CR

Accepted article by IEEE International Conference on Communications  (ICC 2024), Copyright 2024 IEEE

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06388v1) [paper-pdf](http://arxiv.org/pdf/2403.06388v1)

**Authors**: Md. Shirajum Munir, Sravanthi Proddatoori, Manjushree Muralidhara, Walid Saad, Zhu Han, Sachin Shetty

**Abstract**: Understanding the potential of generative AI (GenAI)-based attacks on the power grid is a fundamental challenge that must be addressed in order to protect the power grid by realizing and validating risk in new attack vectors. In this paper, a novel zero trust framework for a power grid supply chain (PGSC) is proposed. This framework facilitates early detection of potential GenAI-driven attack vectors (e.g., replay and protocol-type attacks), assessment of tail risk-based stability measures, and mitigation of such threats. First, a new zero trust system model of PGSC is designed and formulated as a zero-trust problem that seeks to guarantee for a stable PGSC by realizing and defending against GenAI-driven cyber attacks. Second, in which a domain-specific generative adversarial networks (GAN)-based attack generation mechanism is developed to create a new vulnerability cyberspace for further understanding that threat. Third, tail-based risk realization metrics are developed and implemented for quantifying the extreme risk of a potential attack while leveraging a trust measurement approach for continuous validation. Fourth, an ensemble learning-based bootstrap aggregation scheme is devised to detect the attacks that are generating synthetic identities with convincing user and distributed energy resources device profiles. Experimental results show the efficacy of the proposed zero trust framework that achieves an accuracy of 95.7% on attack vector generation, a risk measure of 9.61% for a 95% stable PGSC, and a 99% confidence in defense against GenAI-driven attack.



## **15. Towards Scalable and Robust Model Versioning**

cs.LG

Published in IEEE SaTML 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2401.09574v2) [paper-pdf](http://arxiv.org/pdf/2401.09574v2)

**Authors**: Wenxin Ding, Arjun Nitin Bhagoji, Ben Y. Zhao, Haitao Zheng

**Abstract**: As the deployment of deep learning models continues to expand across industries, the threat of malicious incursions aimed at gaining access to these deployed models is on the rise. Should an attacker gain access to a deployed model, whether through server breaches, insider attacks, or model inversion techniques, they can then construct white-box adversarial attacks to manipulate the model's classification outcomes, thereby posing significant risks to organizations that rely on these models for critical tasks. Model owners need mechanisms to protect themselves against such losses without the necessity of acquiring fresh training data - a process that typically demands substantial investments in time and capital.   In this paper, we explore the feasibility of generating multiple versions of a model that possess different attack properties, without acquiring new training data or changing model architecture. The model owner can deploy one version at a time and replace a leaked version immediately with a new version. The newly deployed model version can resist adversarial attacks generated leveraging white-box access to one or all previously leaked versions. We show theoretically that this can be accomplished by incorporating parameterized hidden distributions into the model training data, forcing the model to learn task-irrelevant features uniquely defined by the chosen data. Additionally, optimal choices of hidden distributions can produce a sequence of model versions capable of resisting compound transferability attacks over time. Leveraging our analytical insights, we design and implement a practical model versioning method for DNN classifiers, which leads to significant robustness improvements over existing methods. We believe our work presents a promising direction for safeguarding DNN services beyond their initial deployment.



## **16. Fake or Compromised? Making Sense of Malicious Clients in Federated Learning**

cs.LG

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2403.06319v1) [paper-pdf](http://arxiv.org/pdf/2403.06319v1)

**Authors**: Hamid Mozaffari, Sunav Choudhary, Amir Houmansadr

**Abstract**: Federated learning (FL) is a distributed machine learning paradigm that enables training models on decentralized data. The field of FL security against poisoning attacks is plagued with confusion due to the proliferation of research that makes different assumptions about the capabilities of adversaries and the adversary models they operate under. Our work aims to clarify this confusion by presenting a comprehensive analysis of the various poisoning attacks and defensive aggregation rules (AGRs) proposed in the literature, and connecting them under a common framework. To connect existing adversary models, we present a hybrid adversary model, which lies in the middle of the spectrum of adversaries, where the adversary compromises a few clients, trains a generative (e.g., DDPM) model with their compromised samples, and generates new synthetic data to solve an optimization for a stronger (e.g., cheaper, more practical) attack against different robust aggregation rules. By presenting the spectrum of FL adversaries, we aim to provide practitioners and researchers with a clear understanding of the different types of threats they need to consider when designing FL systems, and identify areas where further research is needed.



## **17. Improving behavior based authentication against adversarial attack using XAI**

cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2402.16430v2) [paper-pdf](http://arxiv.org/pdf/2402.16430v2)

**Authors**: Dong Qin, George Amariucai, Daji Qiao, Yong Guan

**Abstract**: In recent years, machine learning models, especially deep neural networks, have been widely used for classification tasks in the security domain. However, these models have been shown to be vulnerable to adversarial manipulation: small changes learned by an adversarial attack model, when applied to the input, can cause significant changes in the output. Most research on adversarial attacks and corresponding defense methods focuses only on scenarios where adversarial samples are directly generated by the attack model. In this study, we explore a more practical scenario in behavior-based authentication, where adversarial samples are collected from the attacker. The generated adversarial samples from the model are replicated by attackers with a certain level of discrepancy. We propose an eXplainable AI (XAI) based defense strategy against adversarial attacks in such scenarios. A feature selector, trained with our method, can be used as a filter in front of the original authenticator. It filters out features that are more vulnerable to adversarial attacks or irrelevant to authentication, while retaining features that are more robust. Through comprehensive experiments, we demonstrate that our XAI based defense strategy is effective against adversarial attacks and outperforms other defense strategies, such as adversarial training and defensive distillation.



## **18. Learn from the Past: A Proxy Guided Adversarial Defense Framework with Self Distillation Regularization**

cs.LG

13 Pages

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.12713v2) [paper-pdf](http://arxiv.org/pdf/2310.12713v2)

**Authors**: Yaohua Liu, Jiaxin Gao, Xianghao Jiao, Zhu Liu, Xin Fan, Risheng Liu

**Abstract**: Adversarial Training (AT), pivotal in fortifying the robustness of deep learning models, is extensively adopted in practical applications. However, prevailing AT methods, relying on direct iterative updates for target model's defense, frequently encounter obstacles such as unstable training and catastrophic overfitting. In this context, our work illuminates the potential of leveraging the target model's historical states as a proxy to provide effective initialization and defense prior, which results in a general proxy guided defense framework, `LAST' ({\bf L}earn from the P{\bf ast}). Specifically, LAST derives response of the proxy model as dynamically learned fast weights, which continuously corrects the update direction of the target model. Besides, we introduce a self-distillation regularized defense objective, ingeniously designed to steer the proxy model's update trajectory without resorting to external teacher models, thereby ameliorating the impact of catastrophic overfitting on performance. Extensive experiments and ablation studies showcase the framework's efficacy in markedly improving model robustness (e.g., up to 9.2\% and 20.3\% enhancement in robust accuracy on CIFAR10 and CIFAR100 datasets, respectively) and training stability. These improvements are consistently observed across various model architectures, larger datasets, perturbation sizes, and attack modalities, affirming LAST's ability to consistently refine both single-step and multi-step AT strategies. The code will be available at~\url{https://github.com/callous-youth/LAST}.



## **19. Deep Reinforcement Learning with Spiking Q-learning**

cs.NE

15 pages, 7 figures

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2201.09754v2) [paper-pdf](http://arxiv.org/pdf/2201.09754v2)

**Authors**: Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian

**Abstract**: With the help of special neuromorphic hardware, spiking neural networks (SNNs) are expected to realize artificial intelligence (AI) with less energy consumption. It provides a promising energy-efficient way for realistic control tasks by combining SNNs with deep reinforcement learning (RL). There are only a few existing SNN-based RL methods at present. Most of them either lack generalization ability or employ Artificial Neural Networks (ANNs) to estimate value function in training. The former needs to tune numerous hyper-parameters for each scenario, and the latter limits the application of different types of RL algorithm and ignores the large energy consumption in training. To develop a robust spike-based RL method, we draw inspiration from non-spiking interneurons found in insects and propose the deep spiking Q-network (DSQN), using the membrane voltage of non-spiking neurons as the representation of Q-value, which can directly learn robust policies from high-dimensional sensory inputs using end-to-end RL. Experiments conducted on 17 Atari games demonstrate the DSQN is effective and even outperforms the ANN-based deep Q-network (DQN) in most games. Moreover, the experiments show superior learning stability and robustness to adversarial attacks of DSQN.



## **20. Language-Driven Anchors for Zero-Shot Adversarial Robustness**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2301.13096v3) [paper-pdf](http://arxiv.org/pdf/2301.13096v3)

**Authors**: Xiao Li, Wei Zhang, Yining Liu, Zhanhao Hu, Bo Zhang, Xiaolin Hu

**Abstract**: Deep Neural Networks (DNNs) are known to be susceptible to adversarial attacks. Previous researches mainly focus on improving adversarial robustness in the fully supervised setting, leaving the challenging domain of zero-shot adversarial robustness an open question. In this work, we investigate this domain by leveraging the recent advances in large vision-language models, such as CLIP, to introduce zero-shot adversarial robustness to DNNs. We propose LAAT, a Language-driven, Anchor-based Adversarial Training strategy. LAAT utilizes the features of a text encoder for each category as fixed anchors (normalized feature embeddings) for each category, which are then employed for adversarial training. By leveraging the semantic consistency of the text encoders, LAAT aims to enhance the adversarial robustness of the image model on novel categories. However, naively using text encoders leads to poor results. Through analysis, we identified the issue to be the high cosine similarity between text encoders. We then design an expansion algorithm and an alignment cross-entropy loss to alleviate the problem. Our experimental results demonstrated that LAAT significantly improves zero-shot adversarial robustness over state-of-the-art methods. LAAT has the potential to enhance adversarial robustness by large-scale multimodal models, especially when labeled data is unavailable during training.



## **21. Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**

cs.CV

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2401.16352v2) [paper-pdf](http://arxiv.org/pdf/2401.16352v2)

**Authors**: Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline called Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks resulting in the robustness generalization to unseen attacks and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves state-of-the-art results and exhibits generalization ability against unseen attacks.



## **22. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.19181v2) [paper-pdf](http://arxiv.org/pdf/2310.19181v2)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs, i.e., ChatGPT (GPT 3.5 Turbo), GPT 4, Claude, and Bard, to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing websites and emails that can convincingly imitate well-known brands and also deploy a range of evasive tactics that are used to elude detection mechanisms employed by anti-phishing systems. These attacks can be generated using unmodified or "vanilla" versions of these LLMs without requiring any prior adversarial exploits such as jailbreaking. We evaluate the performance of the LLMs towards generating these attacks and find that they can also be utilized to create malicious prompts that, in turn, can be fed back to the model to generate phishing scams - thus massively reducing the prompt-engineering effort required by attackers to scale these threats. As a countermeasure, we build a BERT-based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content. Our model is transferable across all four commercial LLMs, attaining an average accuracy of 96% for phishing website prompts and 94% for phishing email prompts. We also disclose the vulnerabilities to the concerned LLMs, with Google acknowledging it as a severe issue. Our detection model is available for use at Hugging Face, as well as a ChatGPT Actions plugin.



## **23. Hard-label based Small Query Black-box Adversarial Attack**

cs.LG

11 pages, 3 figures

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2403.06014v1) [paper-pdf](http://arxiv.org/pdf/2403.06014v1)

**Authors**: Jeonghwan Park, Paul Miller, Niall McLaughlin

**Abstract**: We consider the hard label based black box adversarial attack setting which solely observes predicted classes from the target model. Most of the attack methods in this setting suffer from impractical number of queries required to achieve a successful attack. One approach to tackle this drawback is utilising the adversarial transferability between white box surrogate models and black box target model. However, the majority of the methods adopting this approach are soft label based to take the full advantage of zeroth order optimisation. Unlike mainstream methods, we propose a new practical setting of hard label based attack with an optimisation process guided by a pretrained surrogate model. Experiments show the proposed method significantly improves the query efficiency of the hard label based black-box attack across various target model architectures. We find the proposed method achieves approximately 5 times higher attack success rate compared to the benchmarks, especially at the small query budgets as 100 and 250.



## **24. IOI: Invisible One-Iteration Adversarial Attack on No-Reference Image- and Video-Quality Metrics**

eess.IV

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2403.05955v1) [paper-pdf](http://arxiv.org/pdf/2403.05955v1)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: No-reference image- and video-quality metrics are widely used in video processing benchmarks. The robustness of learning-based metrics under video attacks has not been widely studied. In addition to having success, attacks that can be employed in video processing benchmarks must be fast and imperceptible. This paper introduces an Invisible One-Iteration (IOI) adversarial attack on no reference image and video quality metrics. We compared our method alongside eight prior approaches using image and video datasets via objective and subjective tests. Our method exhibited superior visual quality across various attacked metric architectures while maintaining comparable attack success and speed. We made the code available on GitHub.



## **25. SoK: Secure Human-centered Wireless Sensing**

cs.CR

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2211.12087v2) [paper-pdf](http://arxiv.org/pdf/2211.12087v2)

**Authors**: Wei Sun, Tingjun Chen, Neil Gong

**Abstract**: Human-centered wireless sensing (HCWS) aims to understand the fine-grained environment and activities of a human using the diverse wireless signals around him/her. While the sensed information about a human can be used for many good purposes such as enhancing life quality, an adversary can also abuse it to steal private information about the human (e.g., location and person's identity). However, the literature lacks a systematic understanding of the privacy vulnerabilities of wireless sensing and the defenses against them, resulting in the privacy-compromising HCWS design.   In this work, we aim to bridge this gap to achieve the vision of secure human-centered wireless sensing. First, we propose a signal processing pipeline to identify private information leakage and further understand the benefits and tradeoffs of wireless sensing-based inference attacks and defenses. Based on this framework, we present the taxonomy of existing inference attacks and defenses. As a result, we can identify the open challenges and gaps in achieving privacy-preserving human-centered wireless sensing in the era of machine learning and further propose directions for future research in this field.



## **26. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

cs.RO

8 pages (7 content, 1 reference). 5 figures, submitted to the IEEE  Robotics and Automation Letters (RA-L)

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05666v1) [paper-pdf](http://arxiv.org/pdf/2403.05666v1)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method to assess the resilience of the Iterative Closest Point (ICP) algorithm via deep-learning-based attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms prior to deployments is of utmost importance. The ICP algorithm has become the standard for lidar-based localization. However, the pose estimate it produces can be greatly affected by corruption in the measurements. Corruption can arise from a variety of scenarios such as occlusions, adverse weather, or mechanical issues in the sensor. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP empirically, our method focuses on finding the maximum possible ICP pose error using perturbation-based adversarial attacks. The proposed attack induces significant pose errors on ICP and outperforms baselines more than 88% of the time across a wide range of scenarios. As an example application, we demonstrate that our attack can be used to identify areas on a map where ICP is particularly vulnerable to corruption in the measurements.



## **27. Invariant Aggregator for Defending against Federated Backdoor Attacks**

cs.LG

AISTATS 2024 camera-ready

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2210.01834v4) [paper-pdf](http://arxiv.org/pdf/2210.01834v4)

**Authors**: Xiaoyang Wang, Dimitrios Dimitriadis, Sanmi Koyejo, Shruti Tople

**Abstract**: Federated learning enables training high-utility models across several clients without directly sharing their private data. As a downside, the federated setting makes the model vulnerable to various adversarial attacks in the presence of malicious clients. Despite the theoretical and empirical success in defending against attacks that aim to degrade models' utility, defense against backdoor attacks that increase model accuracy on backdoor samples exclusively without hurting the utility on other samples remains challenging. To this end, we first analyze the failure modes of existing defenses over a flat loss landscape, which is common for well-designed neural networks such as Resnet (He et al., 2015) but is often overlooked by previous works. Then, we propose an invariant aggregator that redirects the aggregated update to invariant directions that are generally useful via selectively masking out the update elements that favor few and possibly malicious clients. Theoretical results suggest that our approach provably mitigates backdoor attacks and remains effective over flat loss landscapes. Empirical results on three datasets with different modalities and varying numbers of clients further demonstrate that our approach mitigates a broad class of backdoor attacks with a negligible cost on the model utility.



## **28. Can LLMs Follow Simple Rules?**

cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules;  revised content

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2311.04235v3) [paper-pdf](http://arxiv.org/pdf/2311.04235v3)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Basel Alomair, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.



## **29. On Practicality of Using ARM TrustZone Trusted Execution Environment for Securing Programmable Logic Controllers**

cs.CR

To appear at ACM AsiaCCS 2024

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05448v1) [paper-pdf](http://arxiv.org/pdf/2403.05448v1)

**Authors**: Zhiang Li, Daisuke Mashima, Wen Shei Ong, Ertem Esiner, Zbigniew Kalbarczyk, Ee-Chien Chang

**Abstract**: Programmable logic controllers (PLCs) are crucial devices for implementing automated control in various industrial control systems (ICS), such as smart power grids, water treatment systems, manufacturing, and transportation systems. Owing to their importance, PLCs are often the target of cyber attackers that are aiming at disrupting the operation of ICS, including the nation's critical infrastructure, by compromising the integrity of control logic execution. While a wide range of cybersecurity solutions for ICS have been proposed, they cannot counter strong adversaries with a foothold on the PLC devices, which could manipulate memory, I/O interface, or PLC logic itself. These days, many ICS devices in the market, including PLCs, run on ARM-based processors, and there is a promising security technology called ARM TrustZone, to offer a Trusted Execution Environment (TEE) on embedded devices. Envisioning that such a hardware-assisted security feature becomes available for ICS devices in the near future, this paper investigates the application of the ARM TrustZone TEE technology for enhancing the security of PLC. Our aim is to evaluate the feasibility and practicality of the TEE-based PLCs through the proof-of-concept design and implementation using open-source software such as OP-TEE and OpenPLC. Our evaluation assesses the performance and resource consumption in real-world ICS configurations, and based on the results, we discuss bottlenecks in the OP-TEE secure OS towards a large-scale ICS and desired changes for its application on ICS devices. Our implementation is made available to public for further study and research.



## **30. EVD4UAV: An Altitude-Sensitive Benchmark to Evade Vehicle Detection in UAV**

cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05422v1) [paper-pdf](http://arxiv.org/pdf/2403.05422v1)

**Authors**: Huiming Sun, Jiacheng Guo, Zibo Meng, Tianyun Zhang, Jianwu Fang, Yuewei Lin, Hongkai Yu

**Abstract**: Vehicle detection in Unmanned Aerial Vehicle (UAV) captured images has wide applications in aerial photography and remote sensing. There are many public benchmark datasets proposed for the vehicle detection and tracking in UAV images. Recent studies show that adding an adversarial patch on objects can fool the well-trained deep neural networks based object detectors, posing security concerns to the downstream tasks. However, the current public UAV datasets might ignore the diverse altitudes, vehicle attributes, fine-grained instance-level annotation in mostly side view with blurred vehicle roof, so none of them is good to study the adversarial patch based vehicle detection attack problem. In this paper, we propose a new dataset named EVD4UAV as an altitude-sensitive benchmark to evade vehicle detection in UAV with 6,284 images and 90,886 fine-grained annotated vehicles. The EVD4UAV dataset has diverse altitudes (50m, 70m, 90m), vehicle attributes (color, type), fine-grained annotation (horizontal and rotated bounding boxes, instance-level mask) in top view with clear vehicle roof. One white-box and two black-box patch based attack methods are implemented to attack three classic deep neural networks based object detectors on EVD4UAV. The experimental results show that these representative attack methods could not achieve the robust altitude-insensitive attack performance.



## **31. The Impact of Quantization on the Robustness of Transformer-based Text Classifiers**

cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05365v1) [paper-pdf](http://arxiv.org/pdf/2403.05365v1)

**Authors**: Seyed Parsa Neshaei, Yasaman Boreshban, Gholamreza Ghassem-Sani, Seyed Abolghasem Mirroshandel

**Abstract**: Transformer-based models have made remarkable advancements in various NLP areas. Nevertheless, these models often exhibit vulnerabilities when confronted with adversarial attacks. In this paper, we explore the effect of quantization on the robustness of Transformer-based models. Quantization usually involves mapping a high-precision real number to a lower-precision value, aiming at reducing the size of the model at hand. To the best of our knowledge, this work is the first application of quantization on the robustness of NLP models. In our experiments, we evaluate the impact of quantization on BERT and DistilBERT models in text classification using SST-2, Emotion, and MR datasets. We also evaluate the performance of these models against TextFooler, PWWS, and PSO adversarial attacks. Our findings show that quantization significantly improves (by an average of 18.68%) the adversarial accuracy of the models. Furthermore, we compare the effect of quantization versus that of the adversarial training approach on robustness. Our experiments indicate that quantization increases the robustness of the model by 18.80% on average compared to adversarial training without imposing any extra computational overhead during training. Therefore, our results highlight the effectiveness of quantization in improving the robustness of NLP models.



## **32. Hide in Thicket: Generating Imperceptible and Rational Adversarial Perturbations on 3D Point Clouds**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05247v1) [paper-pdf](http://arxiv.org/pdf/2403.05247v1)

**Authors**: Tianrui Lou, Xiaojun Jia, Jindong Gu, Li Liu, Siyuan Liang, Bangyan He, Xiaochun Cao

**Abstract**: Adversarial attack methods based on point manipulation for 3D point cloud classification have revealed the fragility of 3D models, yet the adversarial examples they produce are easily perceived or defended against. The trade-off between the imperceptibility and adversarial strength leads most point attack methods to inevitably introduce easily detectable outlier points upon a successful attack. Another promising strategy, shape-based attack, can effectively eliminate outliers, but existing methods often suffer significant reductions in imperceptibility due to irrational deformations. We find that concealing deformation perturbations in areas insensitive to human eyes can achieve a better trade-off between imperceptibility and adversarial strength, specifically in parts of the object surface that are complex and exhibit drastic curvature changes. Therefore, we propose a novel shape-based adversarial attack method, HiT-ADV, which initially conducts a two-stage search for attack regions based on saliency and imperceptibility scores, and then adds deformation perturbations in each attack region using Gaussian kernel functions. Additionally, HiT-ADV is extendable to physical attack. We propose that by employing benign resampling and benign rigid transformations, we can further enhance physical adversarial strength with little sacrifice to imperceptibility. Extensive experiments have validated the superiority of our method in terms of adversarial and imperceptible properties in both digital and physical spaces. Our code is avaliable at: https://github.com/TRLou/HiT-ADV.



## **33. Adversarial Sparse Teacher: Defense Against Distillation-Based Model Stealing Attacks Using Adversarial Examples**

cs.LG

12 pages, 3 figures, 6 tables

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05181v1) [paper-pdf](http://arxiv.org/pdf/2403.05181v1)

**Authors**: Eda Yilmaz, Hacer Yalim Keles

**Abstract**: Knowledge Distillation (KD) facilitates the transfer of discriminative capabilities from an advanced teacher model to a simpler student model, ensuring performance enhancement without compromising accuracy. It is also exploited for model stealing attacks, where adversaries use KD to mimic the functionality of a teacher model. Recent developments in this domain have been influenced by the Stingy Teacher model, which provided empirical analysis showing that sparse outputs can significantly degrade the performance of student models. Addressing the risk of intellectual property leakage, our work introduces an approach to train a teacher model that inherently protects its logits, influenced by the Nasty Teacher concept. Differing from existing methods, we incorporate sparse outputs of adversarial examples with standard training data to strengthen the teacher's defense against student distillation. Our approach carefully reduces the relative entropy between the original and adversarially perturbed outputs, allowing the model to produce adversarial logits with minimal impact on overall performance. The source codes will be made publicly available soon.



## **34. Warfare:Breaking the Watermark Protection of AI-Generated Content**

cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2310.07726v3) [paper-pdf](http://arxiv.org/pdf/2310.07726v3)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.



## **35. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2312.14197v3) [paper-pdf](http://arxiv.org/pdf/2312.14197v3)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models (LLMs) with external content has enabled more up-to-date and wide-ranging applications of LLMs, such as Microsoft Copilot. However, this integration has also exposed LLMs to the risk of indirect prompt injection attacks, where an attacker can embed malicious instructions within external content, compromising LLM output and causing responses to deviate from user expectations. To investigate this important but underexplored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to evaluate the risk of such attacks. Based on the evaluation, our work makes a key analysis of the underlying reason for the success of the attack, namely the inability of LLMs to distinguish between instructions and external content and the absence of LLMs' awareness to not execute instructions within external content. Building upon this analysis, we develop two black-box methods based on prompt learning and a white-box defense method based on fine-tuning with adversarial training accordingly. Experimental results demonstrate that black-box defenses are highly effective in mitigating these attacks, while the white-box defense reduces the attack success rate to near-zero levels. Overall, our work systematically investigates indirect prompt injection attacks by introducing a benchmark, analyzing the underlying reason for the success of the attack, and developing an initial set of defenses.



## **36. Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume**

cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05100v1) [paper-pdf](http://arxiv.org/pdf/2403.05100v1)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Zhiyuan Yang, Qingfu Zhang

**Abstract**: The escalating threat of adversarial attacks on deep learning models, particularly in security-critical fields, has underscored the need for robust deep learning systems. Conventional robustness evaluations have relied on adversarial accuracy, which measures a model's performance under a specific perturbation intensity. However, this singular metric does not fully encapsulate the overall resilience of a model against varying degrees of perturbation. To address this gap, we propose a new metric termed adversarial hypervolume, assessing the robustness of deep learning models comprehensively over a range of perturbation intensities from a multi-objective optimization standpoint. This metric allows for an in-depth comparison of defense mechanisms and recognizes the trivial improvements in robustness afforded by less potent defensive strategies. Additionally, we adopt a novel training algorithm that enhances adversarial robustness uniformly across various perturbation intensities, in contrast to methods narrowly focused on optimizing adversarial accuracy. Our extensive empirical studies validate the effectiveness of the adversarial hypervolume metric, demonstrating its ability to reveal subtle differences in robustness that adversarial accuracy overlooks. This research contributes a new measure of robustness and establishes a standard for assessing and benchmarking the resilience of current and future defensive models against adversarial threats.



## **37. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05030v1) [paper-pdf](http://arxiv.org/pdf/2403.05030v1)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: AI systems sometimes exhibit harmful unintended behaviors post-deployment. This is often despite extensive diagnostics and debugging by developers. Minimizing risks from models is challenging because the attack surface is so large. It is not tractable to exhaustively search for inputs that may cause a model to fail. Red-teaming and adversarial training (AT) are commonly used to make AI systems more robust. However, they have not been sufficient to avoid many real-world failure modes that differ from the ones adversarially trained on. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.



## **38. Optimal Denial-of-Service Attacks Against Status Updating**

cs.IT

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04489v1) [paper-pdf](http://arxiv.org/pdf/2403.04489v1)

**Authors**: Saad Kriouile, Mohamad Assaad, Deniz Gündüz, Touraj Soleymani

**Abstract**: In this paper, we investigate denial-of-service attacks against status updating. The target system is modeled by a Markov chain and an unreliable wireless channel, and the performance of status updating in the target system is measured based on two metrics: age of information and age of incorrect information. Our objective is to devise optimal attack policies that strike a balance between the deterioration of the system's performance and the adversary's energy. We model the optimal problem as a Markov decision process and prove rigorously that the optimal jamming policy is a threshold-based policy under both metrics. In addition, we provide a low-complexity algorithm to obtain the optimal threshold value of the jamming policy. Our numerical results show that the networked system with the age-of-incorrect-information metric is less sensitive to jamming attacks than with the age-of-information metric. Index Terms-age of incorrect information, age of information, cyber-physical systems, status updating, remote monitoring.



## **39. AdvQuNN: A Methodology for Analyzing the Adversarial Robustness of Quanvolutional Neural Networks**

quant-ph

7 pages, 6 figures

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.05596v1) [paper-pdf](http://arxiv.org/pdf/2403.05596v1)

**Authors**: Walid El Maouaki, Alberto Marchisio, Taoufik Said, Mohamed Bennai, Muhammad Shafique

**Abstract**: Recent advancements in quantum computing have led to the development of hybrid quantum neural networks (HQNNs) that employ a mixed set of quantum layers and classical layers, such as Quanvolutional Neural Networks (QuNNs). While several works have shown security threats of classical neural networks, such as adversarial attacks, their impact on QuNNs is still relatively unexplored. This work tackles this problem by designing AdvQuNN, a specialized methodology to investigate the robustness of HQNNs like QuNNs against adversarial attacks. It employs different types of Ansatzes as parametrized quantum circuits and different types of adversarial attacks. This study aims to rigorously assess the influence of quantum circuit architecture on the resilience of QuNN models, which opens up new pathways for enhancing the robustness of QuNNs and advancing the field of quantum cybersecurity. Our results show that, compared to classical convolutional networks, QuNNs achieve up to 60\% higher robustness for the MNIST and 40\% for FMNIST datasets.



## **40. Pilot Spoofing Attack on the Downlink of Cell-Free Massive MIMO: From the Perspective of Adversaries**

cs.IT

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04435v1) [paper-pdf](http://arxiv.org/pdf/2403.04435v1)

**Authors**: Weiyang Xu, Yuan Zhang, Ruiguang Wang, Hien Quoc Ngo, Wei Xiang

**Abstract**: The channel hardening effect is less pronounced in the cell-free massive multiple-input multiple-output (mMIMO) system compared to its cellular counterpart, making it necessary to estimate the downlink effective channel gains to ensure decent performance. However, the downlink training inadvertently creates an opportunity for adversarial nodes to launch pilot spoofing attacks (PSAs). First, we demonstrate that adversarial distributed access points (APs) can severely degrade the achievable downlink rate. They achieve this by estimating their channels to users in the uplink training phase and then precoding and sending the same pilot sequences as those used by legitimate APs during the downlink training phase. Then, the impact of the downlink PSA is investigated by rigorously deriving a closed-form expression of the per-user achievable downlink rate. By employing the min-max criterion to optimize the power allocation coefficients, the maximum per-user achievable rate of downlink transmission is minimized from the perspective of adversarial APs. As an alternative to the downlink PSA, adversarial APs may opt to precode random interference during the downlink data transmission phase in order to disrupt legitimate communications. In this scenario, the achievable downlink rate is derived, and then power optimization algorithms are also developed. We present numerical results to showcase the detrimental impact of the downlink PSA and compare the effects of these two types of attacks.



## **41. Evaluating the security of CRYSTALS-Dilithium in the quantum random oracle model**

cs.CR

23 pages; v2: added description of CRYSTALS-Dilithium, improved  analysis of concrete parameters

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2312.16619v2) [paper-pdf](http://arxiv.org/pdf/2312.16619v2)

**Authors**: Kelsey A. Jackson, Carl A. Miller, Daochen Wang

**Abstract**: In the wake of recent progress on quantum computing hardware, the National Institute of Standards and Technology (NIST) is standardizing cryptographic protocols that are resistant to attacks by quantum adversaries. The primary digital signature scheme that NIST has chosen is CRYSTALS-Dilithium. The hardness of this scheme is based on the hardness of three computational problems: Module Learning with Errors (MLWE), Module Short Integer Solution (MSIS), and SelfTargetMSIS. MLWE and MSIS have been well-studied and are widely believed to be secure. However, SelfTargetMSIS is novel and, though classically as hard as MSIS, its quantum hardness is unclear. In this paper, we provide the first proof of the hardness of SelfTargetMSIS via a reduction from MLWE in the Quantum Random Oracle Model (QROM). Our proof uses recently developed techniques in quantum reprogramming and rewinding. A central part of our approach is a proof that a certain hash function, derived from the MSIS problem, is collapsing. From this approach, we deduce a new security proof for Dilithium under appropriate parameter settings. Compared to the previous work by Kiltz, Lyubashevsky, and Schaffner (EUROCRYPT 2018) that gave the only other rigorous security proof for a variant of Dilithium, our proof has the advantage of being applicable under the condition q = 1 mod 2n, where q denotes the modulus and n the dimension of the underlying algebraic ring. This condition is part of the original Dilithium proposal and is crucial for the efficient implementation of the scheme. We provide new secure parameter sets for Dilithium under the condition q = 1 mod 2n, finding that our public key size and signature size are about 2.9 times and 1.3 times larger, respectively, than those proposed by Kiltz et al. at the same security level.



## **42. Multi-Agent Reinforcement Learning for Assessing False-Data Injection Attacks on Transportation Networks**

cs.AI

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2312.14625v2) [paper-pdf](http://arxiv.org/pdf/2312.14625v2)

**Authors**: Taha Eghtesad, Sirui Li, Yevgeniy Vorobeychik, Aron Laszka

**Abstract**: The increasing reliance of drivers on navigation applications has made transportation networks more susceptible to data-manipulation attacks by malicious actors. Adversaries may exploit vulnerabilities in the data collection or processing of navigation services to inject false information, and to thus interfere with the drivers' route selection. Such attacks can significantly increase traffic congestions, resulting in substantial waste of time and resources, and may even disrupt essential services that rely on road networks. To assess the threat posed by such attacks, we introduce a computational framework to find worst-case data-injection attacks against transportation networks. First, we devise an adversarial model with a threat actor who can manipulate drivers by increasing the travel times that they perceive on certain roads. Then, we employ hierarchical multi-agent reinforcement learning to find an approximate optimal adversarial strategy for data manipulation. We demonstrate the applicability of our approach through simulating attacks on the Sioux Falls, ND network topology.



## **43. Improving Adversarial Training using Vulnerability-Aware Perturbation Budget**

cs.LG

19 pages, 2 figures

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.04070v1) [paper-pdf](http://arxiv.org/pdf/2403.04070v1)

**Authors**: Olukorede Fakorede, Modeste Atsague, Jin Tian

**Abstract**: Adversarial Training (AT) effectively improves the robustness of Deep Neural Networks (DNNs) to adversarial attacks. Generally, AT involves training DNN models with adversarial examples obtained within a pre-defined, fixed perturbation bound. Notably, individual natural examples from which these adversarial examples are crafted exhibit varying degrees of intrinsic vulnerabilities, and as such, crafting adversarial examples with fixed perturbation radius for all instances may not sufficiently unleash the potency of AT. Motivated by this observation, we propose two simple, computationally cheap vulnerability-aware reweighting functions for assigning perturbation bounds to adversarial examples used for AT, named Margin-Weighted Perturbation Budget (MWPB) and Standard-Deviation-Weighted Perturbation Budget (SDWPB). The proposed methods assign perturbation radii to individual adversarial samples based on the vulnerability of their corresponding natural examples. Experimental results show that the proposed methods yield genuine improvements in the robustness of AT algorithms against various adversarial attacks.



## **44. Improving Adversarial Attacks on Latent Diffusion Model**

cs.CV

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2310.04687v3) [paper-pdf](http://arxiv.org/pdf/2310.04687v3)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu

**Abstract**: Adversarial attacks on Latent Diffusion Model (LDM), the state-of-the-art image generative model, have been adopted as effective protection against malicious finetuning of LDM on unauthorized images. We show that these attacks add an extra error to the score function of adversarial examples predicted by LDM. LDM finetuned on these adversarial examples learns to lower the error by a bias, from which the model is attacked and predicts the score function with biases.   Based on the dynamics, we propose to improve the adversarial attack on LDM by Attacking with Consistent score-function Errors (ACE). ACE unifies the pattern of the extra error added to the predicted score function. This induces the finetuned LDM to learn the same pattern as a bias in predicting the score function. We then introduce a well-crafted pattern to improve the attack. Our method outperforms state-of-the-art methods in adversarial attacks on LDM.



## **45. A Survey on Adversarial Contention Resolution**

cs.DC

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03876v1) [paper-pdf](http://arxiv.org/pdf/2403.03876v1)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.



## **46. Effect of Ambient-Intrinsic Dimension Gap on Adversarial Vulnerability**

cs.LG

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03967v1) [paper-pdf](http://arxiv.org/pdf/2403.03967v1)

**Authors**: Rajdeep Haldar, Yue Xing, Qifan Song

**Abstract**: The existence of adversarial attacks on machine learning models imperceptible to a human is still quite a mystery from a theoretical perspective. In this work, we introduce two notions of adversarial attacks: natural or on-manifold attacks, which are perceptible by a human/oracle, and unnatural or off-manifold attacks, which are not. We argue that the existence of the off-manifold attacks is a natural consequence of the dimension gap between the intrinsic and ambient dimensions of the data. For 2-layer ReLU networks, we prove that even though the dimension gap does not affect generalization performance on samples drawn from the observed data space, it makes the clean-trained model more vulnerable to adversarial perturbations in the off-manifold direction of the data space. Our main results provide an explicit relationship between the $\ell_2,\ell_{\infty}$ attack strength of the on/off-manifold attack and the dimension gap.



## **47. Neural Exec: Learning (and Learning from) Execution Triggers for Prompt Injection Attacks**

cs.CR

v0.1

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03792v1) [paper-pdf](http://arxiv.org/pdf/2403.03792v1)

**Authors**: Dario Pasquini, Martin Strohmeier, Carmela Troncoso

**Abstract**: We introduce a new family of prompt injection attacks, termed Neural Exec. Unlike known attacks that rely on handcrafted strings (e.g., "Ignore previous instructions and..."), we show that it is possible to conceptualize the creation of execution triggers as a differentiable search problem and use learning-based methods to autonomously generate them.   Our results demonstrate that a motivated adversary can forge triggers that are not only drastically more effective than current handcrafted ones but also exhibit inherent flexibility in shape, properties, and functionality. In this direction, we show that an attacker can design and generate Neural Execs capable of persisting through multi-stage preprocessing pipelines, such as in the case of Retrieval-Augmented Generation (RAG)-based applications. More critically, our findings show that attackers can produce triggers that deviate markedly in form and shape from any known attack, sidestepping existing blacklist-based detection and sanitation approaches.



## **48. PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion**

cs.CL

LLM evaluation, Multi-turn, Multi-language, Multi-modal benchmark

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03788v1) [paper-pdf](http://arxiv.org/pdf/2403.03788v1)

**Authors**: Zekai Zhang, Yiduo Guo, Yaobo Liang, Dongyan Zhao, Nan Duan

**Abstract**: The growing dependence on Large Language Models (LLMs) for finishing user instructions necessitates a comprehensive understanding of their robustness to complex task completion in real-world situations. To address this critical need, we propose the PowerPoint Task Completion Robustness benchmark (PPTC-R) to measure LLMs' robustness to the user PPT task instruction and software version. Specifically, we construct adversarial user instructions by attacking user instructions at sentence, semantic, and multi-language levels. To assess the robustness of Language Models to software versions, we vary the number of provided APIs to simulate both the newest version and earlier version settings. Subsequently, we test 3 closed-source and 4 open-source LLMs using a benchmark that incorporates these robustness settings, aiming to evaluate how deviations impact LLMs' API calls for task completion. We find that GPT-4 exhibits the highest performance and strong robustness in our benchmark, particularly in the version update and the multilingual settings. However, we find that all LLMs lose their robustness when confronted with multiple challenges (e.g., multi-turn) simultaneously, leading to significant performance drops. We further analyze the robustness behavior and error reasons of LLMs in our benchmark, which provide valuable insights for researchers to understand the LLM's robustness in task completion and develop more robust LLMs and agents. We release the code and data at \url{https://github.com/ZekaiGalaxy/PPTCR}.



## **49. Verification of Neural Networks' Global Robustness**

cs.LG

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2402.19322v2) [paper-pdf](http://arxiv.org/pdf/2402.19322v2)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstract**: Neural networks are successful in various applications but are also susceptible to adversarial attacks. To show the safety of network classifiers, many verifiers have been introduced to reason about the local robustness of a given input to a given perturbation. While successful, local robustness cannot generalize to unseen inputs. Several works analyze global robustness properties, however, neither can provide a precise guarantee about the cases where a network classifier does not change its classification. In this work, we propose a new global robustness property for classifiers aiming at finding the minimal globally robust bound, which naturally extends the popular local robustness property for classifiers. We introduce VHAGaR, an anytime verifier for computing this bound. VHAGaR relies on three main ideas: encoding the problem as a mixed-integer programming and pruning the search space by identifying dependencies stemming from the perturbation or the network's computation and generalizing adversarial attacks to unknown inputs. We evaluate VHAGaR on several datasets and classifiers and show that, given a three hour timeout, the average gap between the lower and upper bound on the minimal globally robust bound computed by VHAGaR is 1.9, while the gap of an existing global robustness verifier is 154.7. Moreover, VHAGaR is 130.6x faster than this verifier. Our results further indicate that leveraging dependencies and adversarial attacks makes VHAGaR 78.6x faster.



## **50. Simplified PCNet with Robustness**

cs.LG

10 pages, 3 figures

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03676v1) [paper-pdf](http://arxiv.org/pdf/2403.03676v1)

**Authors**: Bingheng Li, Xuanting Xie, Haoxiang Lei, Ruiyi Fang, Zhao Kang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant attention for their success in learning the representation of homophilic or heterophilic graphs. However, they cannot generalize well to real-world graphs with different levels of homophily. In response, the Possion-Charlier Network (PCNet) \cite{li2024pc}, the previous work, allows graph representation to be learned from heterophily to homophily. Although PCNet alleviates the heterophily issue, there remain some challenges in further improving the efficacy and efficiency. In this paper, we simplify PCNet and enhance its robustness. We first extend the filter order to continuous values and reduce its parameters. Two variants with adaptive neighborhood sizes are implemented. Theoretical analysis shows our model's robustness to graph structure perturbations or adversarial attacks. We validate our approach through semi-supervised learning tasks on various datasets representing both homophilic and heterophilic graphs.



