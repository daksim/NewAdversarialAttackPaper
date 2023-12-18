# Latest Adversarial Attack Papers
**update at 2023-12-18 09:51:37**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Coevolutionary Algorithm for Building Robust Decision Trees under Minimax Regret**

cs.LG

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.09078v1) [paper-pdf](http://arxiv.org/pdf/2312.09078v1)

**Authors**: Adam Żychowski, Andrew Perrault, Jacek Mańdziuk

**Abstract**: In recent years, there has been growing interest in developing robust machine learning (ML) models that can withstand adversarial attacks, including one of the most widely adopted, efficient, and interpretable ML algorithms-decision trees (DTs). This paper proposes a novel coevolutionary algorithm (CoEvoRDT) designed to create robust DTs capable of handling noisy high-dimensional data in adversarial contexts. Motivated by the limitations of traditional DT algorithms, we leverage adaptive coevolution to allow DTs to evolve and learn from interactions with perturbed input data. CoEvoRDT alternately evolves competing populations of DTs and perturbed features, enabling construction of DTs with desired properties. CoEvoRDT is easily adaptable to various target metrics, allowing the use of tailored robustness criteria such as minimax regret. Furthermore, CoEvoRDT has potential to improve the results of other state-of-the-art methods by incorporating their outcomes (DTs they produce) into the initial population and optimize them in the process of coevolution. Inspired by the game theory, CoEvoRDT utilizes mixed Nash equilibrium to enhance convergence. The method is tested on 20 popular datasets and shows superior performance compared to 4 state-of-the-art algorithms. It outperformed all competing methods on 13 datasets with adversarial accuracy metrics, and on all 20 considered datasets with minimax regret. Strong experimental results and flexibility in choosing the error measure make CoEvoRDT a promising approach for constructing robust DTs in real-world applications.



## **2. Concealing Sensitive Samples against Gradient Leakage in Federated Learning**

cs.LG

Defence against model inversion attack in federated learning

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2209.05724v2) [paper-pdf](http://arxiv.org/pdf/2209.05724v2)

**Authors**: Jing Wu, Munawar Hayat, Mingyi Zhou, Mehrtash Harandi

**Abstract**: Federated Learning (FL) is a distributed learning paradigm that enhances users privacy by eliminating the need for clients to share raw, private data with the server. Despite the success, recent studies expose the vulnerability of FL to model inversion attacks, where adversaries reconstruct users private data via eavesdropping on the shared gradient information. We hypothesize that a key factor in the success of such attacks is the low entanglement among gradients per data within the batch during stochastic optimization. This creates a vulnerability that an adversary can exploit to reconstruct the sensitive data. Building upon this insight, we present a simple, yet effective defense strategy that obfuscates the gradients of the sensitive data with concealed samples. To achieve this, we propose synthesizing concealed samples to mimic the sensitive data at the gradient level while ensuring their visual dissimilarity from the actual sensitive data. Compared to the previous art, our empirical evaluations suggest that the proposed technique provides the strongest protection while simultaneously maintaining the FL performance.



## **3. DRAM-Locker: A General-Purpose DRAM Protection Mechanism against Adversarial DNN Weight Attacks**

cs.AR

7 pages. arXiv admin note: text overlap with arXiv:2305.08034

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.09027v1) [paper-pdf](http://arxiv.org/pdf/2312.09027v1)

**Authors**: Ranyang Zhou, Sabbir Ahmed, Arman Roohi, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: In this work, we propose DRAM-Locker as a robust general-purpose defense mechanism that can protect DRAM against various adversarial Deep Neural Network (DNN) weight attacks affecting data or page tables. DRAM-Locker harnesses the capabilities of in-DRAM swapping combined with a lock-table to prevent attackers from singling out specific DRAM rows to safeguard DNN's weight parameters. Our results indicate that DRAM-Locker can deliver a high level of protection downgrading the performance of targeted weight attacks to a random attack level. Furthermore, the proposed defense mechanism demonstrates no reduction in accuracy when applied to CIFAR-10 and CIFAR-100. Importantly, DRAM-Locker does not necessitate any software retraining or result in extra hardware burden.



## **4. Amicable Aid: Perturbing Images to Improve Classification Performance**

cs.CV

ICASSP 2023

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2112.04720v4) [paper-pdf](http://arxiv.org/pdf/2112.04720v4)

**Authors**: Juyeop Kim, Jun-Ho Choi, Soobeom Jang, Jong-Seok Lee

**Abstract**: While adversarial perturbation of images to attack deep image classification models pose serious security concerns in practice, this paper suggests a novel paradigm where the concept of image perturbation can benefit classification performance, which we call amicable aid. We show that by taking the opposite search direction of perturbation, an image can be modified to yield higher classification confidence and even a misclassified image can be made correctly classified. This can be also achieved with a large amount of perturbation by which the image is made unrecognizable by human eyes. The mechanism of the amicable aid is explained in the viewpoint of the underlying natural image manifold. Furthermore, we investigate the universal amicable aid, i.e., a fixed perturbation can be applied to multiple images to improve their classification results. While it is challenging to find such perturbations, we show that making the decision boundary as perpendicular to the image manifold as possible via training with modified data is effective to obtain a model for which universal amicable perturbations are more easily found.



## **5. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08793v1) [paper-pdf](http://arxiv.org/pdf/2312.08793v1)

**Authors**: Tony T. Wang, Miles Wang, Kaivu Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .



## **6. AVA: Inconspicuous Attribute Variation-based Adversarial Attack bypassing DeepFake Detection**

cs.CV

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08675v1) [paper-pdf](http://arxiv.org/pdf/2312.08675v1)

**Authors**: Xiangtao Meng, Li Wang, Shanqing Guo, Lei Ju, Qingchuan Zhao

**Abstract**: While DeepFake applications are becoming popular in recent years, their abuses pose a serious privacy threat. Unfortunately, most related detection algorithms to mitigate the abuse issues are inherently vulnerable to adversarial attacks because they are built atop DNN-based classification models, and the literature has demonstrated that they could be bypassed by introducing pixel-level perturbations. Though corresponding mitigation has been proposed, we have identified a new attribute-variation-based adversarial attack (AVA) that perturbs the latent space via a combination of Gaussian prior and semantic discriminator to bypass such mitigation. It perturbs the semantics in the attribute space of DeepFake images, which are inconspicuous to human beings (e.g., mouth open) but can result in substantial differences in DeepFake detection. We evaluate our proposed AVA attack on nine state-of-the-art DeepFake detection algorithms and applications. The empirical results demonstrate that AVA attack defeats the state-of-the-art black box attacks against DeepFake detectors and achieves more than a 95% success rate on two commercial DeepFake detectors. Moreover, our human study indicates that AVA-generated DeepFake images are often imperceptible to humans, which presents huge security and privacy concerns.



## **7. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models**

cs.CR

Version 2 updates: Added comparison of three more evaluation methods  and their reliability check using human labeling. Added results for  jailbreaking Llama2 (individual behavior) and included complexity and  hyperparameter analysis. Revised objectives for prompt leaking. Other minor  changes made

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2310.15140v2) [paper-pdf](http://arxiv.org/pdf/2310.15140v2)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent studies suggest that defending against these attacks is possible: adversarial attacks generate unlimited but unreadable gibberish prompts, detectable by perplexity-based filters; manual jailbreak attacks craft readable prompts, but their limited number due to the necessity of human creativity allows for easy blocking. In this paper, we show that these solutions may be too optimistic. We introduce AutoDAN, an interpretable, gradient-based adversarial attack that merges the strengths of both attack types. Guided by the dual goals of jailbreak and readability, AutoDAN optimizes and generates tokens one by one from left to right, resulting in readable prompts that bypass perplexity filters while maintaining high attack success rates. Notably, these prompts, generated from scratch using gradients, are interpretable and diverse, with emerging strategies commonly seen in manual jailbreak attacks. They also generalize to unforeseen harmful behaviors and transfer to black-box LLMs better than their unreadable counterparts when using limited training data or a single proxy model. Furthermore, we show the versatility of AutoDAN by automatically leaking system prompts using a customized objective. Our work offers a new way to red-team LLMs and understand jailbreak mechanisms via interpretability.



## **8. Towards Inductive Robustness: Distilling and Fostering Wave-induced Resonance in Transductive GCNs Against Graph Adversarial Attacks**

cs.LG

AAAI 2024

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08651v1) [paper-pdf](http://arxiv.org/pdf/2312.08651v1)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Hanyuan Huang, Pan Zhou

**Abstract**: Graph neural networks (GNNs) have recently been shown to be vulnerable to adversarial attacks, where slight perturbations in the graph structure can lead to erroneous predictions. However, current robust models for defending against such attacks inherit the transductive limitations of graph convolutional networks (GCNs). As a result, they are constrained by fixed structures and do not naturally generalize to unseen nodes. Here, we discover that transductive GCNs inherently possess a distillable robustness, achieved through a wave-induced resonance process. Based on this, we foster this resonance to facilitate inductive and robust learning. Specifically, we first prove that the signal formed by GCN-driven message passing (MP) is equivalent to the edge-based Laplacian wave, where, within a wave system, resonance can naturally emerge between the signal and its transmitting medium. This resonance provides inherent resistance to malicious perturbations inflicted on the signal system. We then prove that merely three MP iterations within GCNs can induce signal resonance between nodes and edges, manifesting as a coupling between nodes and their distillable surrounding local subgraph. Consequently, we present Graph Resonance-fostering Network (GRN) to foster this resonance via learning node representations from their distilled resonating subgraphs. By capturing the edge-transmitted signals within this subgraph and integrating them with the node signal, GRN embeds these combined signals into the central node's representation. This node-wise embedding approach allows for generalization to unseen nodes. We validate our theoretical findings with experiments, and demonstrate that GRN generalizes robustness to unseen nodes, whilst maintaining state-of-the-art classification accuracy on perturbed graphs.



## **9. Guarding the Grid: Enhancing Resilience in Automated Residential Demand Response Against False Data Injection Attacks**

eess.SY

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08646v1) [paper-pdf](http://arxiv.org/pdf/2312.08646v1)

**Authors**: Thusitha Dayaratne, Carsten Rudolph, Ariel Liebman, Mahsa Salehi

**Abstract**: Utility companies are increasingly leveraging residential demand flexibility and the proliferation of smart/IoT devices to enhance the effectiveness of residential demand response (DR) programs through automated device scheduling. However, the adoption of distributed architectures in these systems exposes them to the risk of false data injection attacks (FDIAs), where adversaries can manipulate decision-making processes by injecting false data. Given the limited control utility companies have over these distributed systems and data, the need for reliable implementations to enhance the resilience of residential DR schemes against FDIAs is paramount. In this work, we present a comprehensive framework that combines DR optimisation, anomaly detection, and strategies for mitigating the impacts of attacks to create a resilient and automated device scheduling system. To validate the robustness of our framework against FDIAs, we performed an evaluation using real-world data sets, highlighting its effectiveness in securing residential DR systems.



## **10. Scalable Ensemble-based Detection Method against Adversarial Attacks for speaker verification**

eess.AS

Submitted to 2024 ICASSP

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08622v1) [paper-pdf](http://arxiv.org/pdf/2312.08622v1)

**Authors**: Haibin Wu, Heng-Cheng Kuo, Yu Tsao, Hung-yi Lee

**Abstract**: Automatic speaker verification (ASV) is highly susceptible to adversarial attacks. Purification modules are usually adopted as a pre-processing to mitigate adversarial noise. However, they are commonly implemented across diverse experimental settings, rendering direct comparisons challenging. This paper comprehensively compares mainstream purification techniques in a unified framework. We find these methods often face a trade-off between user experience and security, as they struggle to simultaneously maintain genuine sample performance and reduce adversarial perturbations. To address this challenge, some efforts have extended purification modules to encompass detection capabilities, aiming to alleviate the trade-off. However, advanced purification modules will always come into the stage to surpass previous detection method. As a result, we further propose an easy-to-follow ensemble approach that integrates advanced purification modules for detection, achieving state-of-the-art (SOTA) performance in countering adversarial noise. Our ensemble method has great potential due to its compatibility with future advanced purification techniques.



## **11. Exploring the Privacy Risks of Adversarial VR Game Design**

cs.CR

Learn more at https://rdi.berkeley.edu/metaverse/metadata

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2207.13176v4) [paper-pdf](http://arxiv.org/pdf/2207.13176v4)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song, James F. O'Brien

**Abstract**: Fifty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Within just a few minutes, an adversarial program had accurately inferred over 25 of their personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. Since the Cambridge Analytica scandal of 2018, adversarially designed gamified elements have been known to constitute a significant privacy threat in conventional social platforms. In this work, we present a case study of how metaverse environments can similarly be adversarially constructed to covertly infer dozens of personal data attributes from seemingly anonymous users. While existing VR privacy research largely focuses on passive observation, we argue that because individuals subconsciously reveal personal information via their motion in response to specific stimuli, active attacks pose an outsized risk in VR environments.



## **12. Defenses in Adversarial Machine Learning: A Survey**

cs.CV

21 pages, 5 figures, 2 tables, 237 reference papers

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08890v1) [paper-pdf](http://arxiv.org/pdf/2312.08890v1)

**Authors**: Baoyuan Wu, Shaokui Wei, Mingli Zhu, Meixi Zheng, Zihao Zhu, Mingda Zhang, Hongrui Chen, Danni Yuan, Li Liu, Qingshan Liu

**Abstract**: Adversarial phenomenon has been widely observed in machine learning (ML) systems, especially in those using deep neural networks, describing that ML systems may produce inconsistent and incomprehensible predictions with humans at some particular cases. This phenomenon poses a serious security threat to the practical application of ML systems, and several advanced attack paradigms have been developed to explore it, mainly including backdoor attacks, weight attacks, and adversarial examples. For each individual attack paradigm, various defense paradigms have been developed to improve the model robustness against the corresponding attack paradigm. However, due to the independence and diversity of these defense paradigms, it is difficult to examine the overall robustness of an ML system against different kinds of attacks.This survey aims to build a systematic review of all existing defense paradigms from a unified perspective. Specifically, from the life-cycle perspective, we factorize a complete machine learning system into five stages, including pre-training, training, post-training, deployment, and inference stages, respectively. Then, we present a clear taxonomy to categorize and review representative defense methods at each individual stage. The unified perspective and presented taxonomies not only facilitate the analysis of the mechanism of each defense paradigm but also help us to understand connections and differences among different defense paradigms, which may inspire future research to develop more advanced, comprehensive defenses.



## **13. Universal Adversarial Framework to Improve Adversarial Robustness for Diabetic Retinopathy Detection**

eess.IV

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08193v1) [paper-pdf](http://arxiv.org/pdf/2312.08193v1)

**Authors**: Samrat Mukherjee, Dibyanayan Bandyopadhyay, Baban Gain, Asif Ekbal

**Abstract**: Diabetic Retinopathy (DR) is a prevalent illness associated with Diabetes which, if left untreated, can result in irreversible blindness. Deep Learning based systems are gradually being introduced as automated support for clinical diagnosis. Since healthcare has always been an extremely important domain demanding error-free performance, any adversaries could pose a big threat to the applicability of such systems. In this work, we use Universal Adversarial Perturbations (UAPs) to quantify the vulnerability of Medical Deep Neural Networks (DNNs) for detecting DR. To the best of our knowledge, this is the very first attempt that works on attacking complete fine-grained classification of DR images using various UAPs. Also, as a part of this work, we use UAPs to fine-tune the trained models to defend against adversarial samples. We experiment on several models and observe that the performance of such models towards unseen adversarial attacks gets boosted on average by $3.41$ Cohen-kappa value and maximum by $31.92$ Cohen-kappa value. The performance degradation on normal data upon ensembling the fine-tuned models was found to be statistically insignificant using t-test, highlighting the benefits of UAP-based adversarial fine-tuning.



## **14. Adversarial Attacks on Graph Neural Networks based Spatial Resource Management in P2P Wireless Communications**

eess.SP

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08181v1) [paper-pdf](http://arxiv.org/pdf/2312.08181v1)

**Authors**: Ahmad Ghasemi, Ehsan Zeraatkar, Majid Moradikia, Seyed, Zekavat

**Abstract**: This paper introduces adversarial attacks targeting a Graph Neural Network (GNN) based radio resource management system in point to point (P2P) communications. Our focus lies on perturbing the trained GNN model during the test phase, specifically targeting its vertices and edges. To achieve this, four distinct adversarial attacks are proposed, each accounting for different constraints, and aiming to manipulate the behavior of the system. The proposed adversarial attacks are formulated as optimization problems, aiming to minimize the system's communication quality. The efficacy of these attacks is investigated against the number of users, signal-to-noise ratio (SNR), and adversary power budget. Furthermore, we address the detection of such attacks from the perspective of the Central Processing Unit (CPU) of the system. To this end, we formulate an optimization problem that involves analyzing the distribution of channel eigenvalues before and after the attacks are applied. This formulation results in a Min-Max optimization problem, allowing us to detect the presence of attacks. Through extensive simulations, we observe that in the absence of adversarial attacks, the eigenvalues conform to Johnson's SU distribution. However, the attacks significantly alter the characteristics of the eigenvalue distribution, and in the most effective attack, they even change the type of the eigenvalue distribution.



## **15. Efficient Representation of the Activation Space in Deep Neural Networks**

cs.LG

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08143v1) [paper-pdf](http://arxiv.org/pdf/2312.08143v1)

**Authors**: Tanya Akumu, Celia Cintas, Girmaw Abebe Tadesse, Adebayo Oshingbesan, Skyler Speakman, Edward McFowland III

**Abstract**: The representations of the activation space of deep neural networks (DNNs) are widely utilized for tasks like natural language processing, anomaly detection and speech recognition. Due to the diverse nature of these tasks and the large size of DNNs, an efficient and task-independent representation of activations becomes crucial. Empirical p-values have been used to quantify the relative strength of an observed node activation compared to activations created by already-known inputs. Nonetheless, keeping raw data for these calculations increases memory resource consumption and raises privacy concerns. To this end, we propose a model-agnostic framework for creating representations of activations in DNNs using node-specific histograms to compute p-values of observed activations without retaining already-known inputs. Our proposed approach demonstrates promising potential when validated with multiple network architectures across various downstream tasks and compared with the kernel density estimates and brute-force empirical baselines. In addition, the framework reduces memory usage by 30% with up to 4 times faster p-value computing time while maintaining state of-the-art detection power in downstream tasks such as the detection of adversarial attacks and synthesized content. Moreover, as we do not persist raw data at inference time, we could potentially reduce susceptibility to attacks and privacy issues.



## **16. Robust Few-Shot Named Entity Recognition with Boundary Discrimination and Correlation Purification**

cs.CL

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07961v1) [paper-pdf](http://arxiv.org/pdf/2312.07961v1)

**Authors**: Xiaojun Xue, Chunxia Zhang, Tianxiang Xu, Zhendong Niu

**Abstract**: Few-shot named entity recognition (NER) aims to recognize novel named entities in low-resource domains utilizing existing knowledge. However, the present few-shot NER models assume that the labeled data are all clean without noise or outliers, and there are few works focusing on the robustness of the cross-domain transfer learning ability to textual adversarial attacks in Few-shot NER. In this work, we comprehensively explore and assess the robustness of few-shot NER models under textual adversarial attack scenario, and found the vulnerability of existing few-shot NER models. Furthermore, we propose a robust two-stage few-shot NER method with Boundary Discrimination and Correlation Purification (BDCP). Specifically, in the span detection stage, the entity boundary discriminative module is introduced to provide a highly distinguishing boundary representation space to detect entity spans. In the entity typing stage, the correlations between entities and contexts are purified by minimizing the interference information and facilitating correlation generalization to alleviate the perturbations caused by textual adversarial attacks. In addition, we construct adversarial examples for few-shot NER based on public datasets Few-NERD and Cross-Dataset. Comprehensive evaluations on those two groups of few-shot NER datasets containing adversarial examples demonstrate the robustness and superiority of the proposed method.



## **17. DifAttack: Query-Efficient Black-Box Attack via Disentangled Feature Space**

cs.CV

Accepted in AAAI'24

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2309.14585v3) [paper-pdf](http://arxiv.org/pdf/2309.14585v3)

**Authors**: Liu Jun, Zhou Jiantao, Zeng Jiandian, Jinyu Tian

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (ASR) and good generalizability. We design a novel attack method based on a Disentangled Feature space, called DifAttack, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack firstly disentangles an image's latent feature into an adversarial feature and a visual feature, where the former dominates the adversarial capability of an image, while the latter largely determines its visual appearance. We train an autoencoder for the disentanglement by using pairs of clean images and their Adversarial Examples (AEs) generated from available surrogate models via white-box attack methods. Eventually, DifAttack iteratively optimizes the adversarial feature according to the query feedback from the victim model until a successful AE is generated, while keeping the visual feature unaltered. In addition, due to the avoidance of using surrogate models' gradient information when optimizing AEs for black-box models, our proposed DifAttack inherently possesses better attack capability in the open-set scenario, where the training dataset of the victim model is unknown. Extensive experimental results demonstrate that our method achieves significant improvements in ASR and query efficiency simultaneously, especially in the targeted attack and open-set scenarios. The code is available at https://github.com/csjunjun/DifAttack.git.



## **18. PromptBench: A Unified Library for Evaluation of Large Language Models**

cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07910v1) [paper-pdf](http://arxiv.org/pdf/2312.07910v1)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.



## **19. Causality Analysis for Evaluating the Security of Large Language Models**

cs.AI

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07876v1) [paper-pdf](http://arxiv.org/pdf/2312.07876v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) such as GPT and Llama2 are increasingly adopted in many safety-critical applications. Their security is thus essential. Even with considerable efforts spent on reinforcement learning from human feedback (RLHF), recent studies have shown that LLMs are still subject to attacks such as adversarial perturbation and Trojan attacks. Further research is thus needed to evaluate their security and/or understand the lack of it. In this work, we propose a framework for conducting light-weight causality-analysis of LLMs at the token, layer, and neuron level. We applied our framework to open-source LLMs such as Llama2 and Vicuna and had multiple interesting discoveries. Based on a layer-level causality analysis, we show that RLHF has the effect of overfitting a model to harmful prompts. It implies that such security can be easily overcome by `unusual' harmful prompts. As evidence, we propose an adversarial perturbation method that achieves 100\% attack success rate on the red-teaming tasks of the Trojan Detection Competition 2023. Furthermore, we show the existence of one mysterious neuron in both Llama2 and Vicuna that has an unreasonably high causal effect on the output. While we are uncertain on why such a neuron exists, we show that it is possible to conduct a ``Trojan'' attack targeting that particular neuron to completely cripple the LLM, i.e., we can generate transferable suffixes to prompts that frequently make the LLM produce meaningless responses.



## **20. Securing Graph Neural Networks in MLaaS: A Comprehensive Realization of Query-based Integrity Verification**

cs.CR

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07870v1) [paper-pdf](http://arxiv.org/pdf/2312.07870v1)

**Authors**: Bang Wu, Xingliang Yuan, Shuo Wang, Qi Li, Minhui Xue, Shirui Pan

**Abstract**: The deployment of Graph Neural Networks (GNNs) within Machine Learning as a Service (MLaaS) has opened up new attack surfaces and an escalation in security concerns regarding model-centric attacks. These attacks can directly manipulate the GNN model parameters during serving, causing incorrect predictions and posing substantial threats to essential GNN applications. Traditional integrity verification methods falter in this context due to the limitations imposed by MLaaS and the distinct characteristics of GNN models.   In this research, we introduce a groundbreaking approach to protect GNN models in MLaaS from model-centric attacks. Our approach includes a comprehensive verification schema for GNN's integrity, taking into account both transductive and inductive GNNs, and accommodating varying pre-deployment knowledge of the models. We propose a query-based verification technique, fortified with innovative node fingerprint generation algorithms. To deal with advanced attackers who know our mechanisms in advance, we introduce randomized fingerprint nodes within our design. The experimental evaluation demonstrates that our method can detect five representative adversarial model-centric attacks, displaying 2 to 4 times greater efficiency compared to baselines.



## **21. SimAC: A Simple Anti-Customization Method against Text-to-Image Synthesis of Diffusion Models**

cs.CV

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07865v1) [paper-pdf](http://arxiv.org/pdf/2312.07865v1)

**Authors**: Feifei Wang, Zhentao Tan, Tianyi Wei, Yue Wu, Qidong Huang

**Abstract**: Despite the success of diffusion-based customization methods on visual content creation, increasing concerns have been raised about such techniques from both privacy and political perspectives. To tackle this issue, several anti-customization methods have been proposed in very recent months, predominantly grounded in adversarial attacks. Unfortunately, most of these methods adopt straightforward designs, such as end-to-end optimization with a focus on adversarially maximizing the original training loss, thereby neglecting nuanced internal properties intrinsic to the diffusion model, and even leading to ineffective optimization in some diffusion time steps. In this paper, we strive to bridge this gap by undertaking a comprehensive exploration of these inherent properties, to boost the performance of current anti-customization approaches. Two aspects of properties are investigated: 1) We examine the relationship between time step selection and the model's perception in the frequency domain of images and find that lower time steps can give much more contributions to adversarial noises. This inspires us to propose an adaptive greedy search for optimal time steps that seamlessly integrates with existing anti-customization methods. 2) We scrutinize the roles of features at different layers during denoising and devise a sophisticated feature-based optimization framework for anti-customization. Experiments on facial benchmarks demonstrate that our approach significantly increases identity disruption, thereby enhancing user privacy and security.



## **22. Radio Signal Classification by Adversarially Robust Quantum Machine Learning**

quant-ph

12 pages, 6 figures

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07821v1) [paper-pdf](http://arxiv.org/pdf/2312.07821v1)

**Authors**: Yanqiu Wu, Eromanga Adermann, Chandra Thapa, Seyit Camtepe, Hajime Suzuki, Muhammad Usman

**Abstract**: Radio signal classification plays a pivotal role in identifying the modulation scheme used in received radio signals, which is essential for demodulation and proper interpretation of the transmitted information. Researchers have underscored the high susceptibility of ML algorithms for radio signal classification to adversarial attacks. Such vulnerability could result in severe consequences, including misinterpretation of critical messages, interception of classified information, or disruption of communication channels. Recent advancements in quantum computing have revolutionized theories and implementations of computation, bringing the unprecedented development of Quantum Machine Learning (QML). It is shown that quantum variational classifiers (QVCs) provide notably enhanced robustness against classical adversarial attacks in image classification. However, no research has yet explored whether QML can similarly mitigate adversarial threats in the context of radio signal classification. This work applies QVCs to radio signal classification and studies their robustness to various adversarial attacks. We also propose the novel application of the approximate amplitude encoding (AAE) technique to encode radio signal data efficiently. Our extensive simulation results present that attacks generated on QVCs transfer well to CNN models, indicating that these adversarial examples can fool neural networks that they are not explicitly designed to attack. However, the converse is not true. QVCs primarily resist the attacks generated on CNNs. Overall, with comprehensive simulations, our results shed new light on the growing field of QML by bridging knowledge gaps in QAML in radio signal classification and uncovering the advantages of applying QML methods in practical applications.



## **23. BarraCUDA: Bringing Electromagnetic Side Channel Into Play to Steal the Weights of Neural Networks from NVIDIA GPUs**

cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07783v1) [paper-pdf](http://arxiv.org/pdf/2312.07783v1)

**Authors**: Peter Horvath, Lukasz Chmielewski, Leo Weissbart, Lejla Batina, Yuval Yarom

**Abstract**: Over the last decade, applications of neural networks have spread to cover all aspects of life. A large number of companies base their businesses on building products that use neural networks for tasks such as face recognition, machine translation, and autonomous cars. They are being used in safety and security-critical applications like high definition maps and medical wristbands, or in globally used products like Google Translate and ChatGPT. Much of the intellectual property underpinning these products is encoded in the exact configuration of the neural networks. Consequently, protecting these is of utmost priority to businesses. At the same time, many of these products need to operate under a strong threat model, in which the adversary has unfettered physical control of the product.   Past work has demonstrated that with physical access, attackers can reverse engineer neural networks that run on scalar microcontrollers, like ARM Cortex M3. However, for performance reasons, neural networks are often implemented on highly-parallel general purpose graphics processing units (GPGPUs), and so far, attacks on these have only recovered course-grained information on the structure of the neural network, but failed to retrieve the weights and biases.   In this work, we present BarraCUDA, a novel attack on GPGPUs that can completely extract the parameters of neural networks. BarraCUDA uses correlation electromagnetic analysis to recover the weights and biases in the convolutional layers of neural networks. We use BarraCUDA to attack the popular NVIDIA Jetson Nano device, demonstrating successful parameter extraction of neural networks in a highly parallel and noisy environment.



## **24. Majority is Not Required: A Rational Analysis of the Private Double-Spend Attack from a Sub-Majority Adversary**

cs.GT

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07709v1) [paper-pdf](http://arxiv.org/pdf/2312.07709v1)

**Authors**: Yanni Georghiades, Rajesh Mishra, Karl Kreder, Sriram Vishwanath

**Abstract**: We study the incentives behind double-spend attacks on Nakamoto-style Proof-of-Work cryptocurrencies. In these systems, miners are allowed to choose which transactions to reference with their block, and a common strategy for selecting transactions is to simply choose those with the highest fees. This can be problematic if these transactions originate from an adversary with substantial (but less than 50\%) computational power, as high-value transactions can present an incentive for a rational adversary to attempt a double-spend attack if they expect to profit. The most common mechanism for deterring double-spend attacks is for the recipients of large transactions to wait for additional block confirmations (i.e., to increase the attack cost). We argue that this defense mechanism is not satisfactory, as the security of the system is contingent on the actions of its users. Instead, we propose that defending against double-spend attacks should be the responsibility of the miners; specifically, miners should limit the amount of transaction value they include in a block (i.e., reduce the attack reward). To this end, we model cryptocurrency mining as a mean-field game in which we augment the standard mining reward function to simulate the presence of a rational, double-spending adversary. We design and implement an algorithm which characterizes the behavior of miners at equilibrium, and we show that miners who use the adversary-aware reward function accumulate more wealth than those who do not. We show that the optimal strategy for honest miners is to limit the amount of value transferred by each block such that the adversary's expected profit is 0. Additionally, we examine Bitcoin's resilience to double-spend attacks. Assuming a 6 block confirmation time, we find that an attacker with at least 25% of the network mining power can expect to profit from a double-spend attack.



## **25. Defending Our Privacy With Backdoors**

cs.LG

14 pages, 10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2310.08320v2) [paper-pdf](http://arxiv.org/pdf/2310.08320v2)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspective on backdoor attacks, but also presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.



## **26. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.04730v2) [paper-pdf](http://arxiv.org/pdf/2312.04730v2)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.



## **27. ReRoGCRL: Representation-based Robustness in Goal-Conditioned Reinforcement Learning**

cs.LG

This paper has been accepted in AAAI24  (https://aaai.org/aaai-conference/)

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07392v1) [paper-pdf](http://arxiv.org/pdf/2312.07392v1)

**Authors**: Xiangyu Yin, Sihao Wu, Jiaxu Liu, Meng Fang, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan

**Abstract**: While Goal-Conditioned Reinforcement Learning (GCRL) has gained attention, its algorithmic robustness, particularly against adversarial perturbations, remains unexplored. Unfortunately, the attacks and robust representation training methods specifically designed for traditional RL are not so effective when applied to GCRL. To address this challenge, we propose the \textit{Semi-Contrastive Representation} attack, a novel approach inspired by the adversarial contrastive attack. Unlike existing attacks in RL, it only necessitates information from the policy function and can be seamlessly implemented during deployment. Furthermore, to mitigate the vulnerability of existing GCRL algorithms, we introduce \textit{Adversarial Representation Tactics}. This strategy combines \textit{Semi-Contrastive Adversarial Augmentation} with \textit{Sensitivity-Aware Regularizer}. It improves the adversarial robustness of the underlying agent against various types of perturbations. Extensive experiments validate the superior performance of our attack and defence mechanism across multiple state-of-the-art GCRL algorithms. Our tool {\bf ReRoGCRL} is available at \url{https://github.com/TrustAI/ReRoGCRL}.



## **28. Eroding Trust In Aerial Imagery: Comprehensive Analysis and Evaluation Of Adversarial Attacks In Geospatial Systems**

cs.CV

Accepted at IEEE AIRP 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07389v1) [paper-pdf](http://arxiv.org/pdf/2312.07389v1)

**Authors**: Michael Lanier, Aayush Dhakal, Zhexiao Xiong, Arthur Li, Nathan Jacobs, Yevgeniy Vorobeychik

**Abstract**: In critical operations where aerial imagery plays an essential role, the integrity and trustworthiness of data are paramount. The emergence of adversarial attacks, particularly those that exploit control over labels or employ physically feasible trojans, threatens to erode that trust, making the analysis and mitigation of these attacks a matter of urgency. We demonstrate how adversarial attacks can degrade confidence in geospatial systems, specifically focusing on scenarios where the attacker's control over labels is restricted and the use of realistic threat vectors. Proposing and evaluating several innovative attack methodologies, including those tailored to overhead images, we empirically show their threat to remote sensing systems using high-quality SpaceNet datasets. Our experimentation reflects the unique challenges posed by aerial imagery, and these preliminary results not only reveal the potential risks but also highlight the non-trivial nature of the problem compared to recent works.



## **29. SSTA: Salient Spatially Transformed Attack**

cs.CV

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07258v1) [paper-pdf](http://arxiv.org/pdf/2312.07258v1)

**Authors**: Renyang Liu, Wei Zhou, Sixin Wu, Jun Zhao, Kwok-Yan Lam

**Abstract**: Extensive studies have demonstrated that deep neural networks (DNNs) are vulnerable to adversarial attacks, which brings a huge security risk to the further application of DNNs, especially for the AI models developed in the real world. Despite the significant progress that has been made recently, existing attack methods still suffer from the unsatisfactory performance of escaping from being detected by naked human eyes due to the formulation of adversarial example (AE) heavily relying on a noise-adding manner. Such mentioned challenges will significantly increase the risk of exposure and result in an attack to be failed. Therefore, in this paper, we propose the Salient Spatially Transformed Attack (SSTA), a novel framework to craft imperceptible AEs, which enhance the stealthiness of AEs by estimating a smooth spatial transform metric on a most critical area to generate AEs instead of adding external noise to the whole image. Compared to state-of-the-art baselines, extensive experiments indicated that SSTA could effectively improve the imperceptibility of the AEs while maintaining a 100\% attack success rate.



## **30. DTA: Distribution Transform-based Attack for Query-Limited Scenario**

cs.CV

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07245v1) [paper-pdf](http://arxiv.org/pdf/2312.07245v1)

**Authors**: Renyang Liu, Wei Zhou, Xin Jin, Song Gao, Yuanyu Wang, Ruxin Wang

**Abstract**: In generating adversarial examples, the conventional black-box attack methods rely on sufficient feedback from the to-be-attacked models by repeatedly querying until the attack is successful, which usually results in thousands of trials during an attack. This may be unacceptable in real applications since Machine Learning as a Service Platform (MLaaS) usually only returns the final result (i.e., hard-label) to the client and a system equipped with certain defense mechanisms could easily detect malicious queries. By contrast, a feasible way is a hard-label attack that simulates an attacked action being permitted to conduct a limited number of queries. To implement this idea, in this paper, we bypass the dependency on the to-be-attacked model and benefit from the characteristics of the distributions of adversarial examples to reformulate the attack problem in a distribution transform manner and propose a distribution transform-based attack (DTA). DTA builds a statistical mapping from the benign example to its adversarial counterparts by tackling the conditional likelihood under the hard-label black-box settings. In this way, it is no longer necessary to query the target model frequently. A well-trained DTA model can directly and efficiently generate a batch of adversarial examples for a certain input, which can be used to attack un-seen models based on the assumed transferability. Furthermore, we surprisingly find that the well-trained DTA model is not sensitive to the semantic spaces of the training dataset, meaning that the model yields acceptable attack performance on other datasets. Extensive experiments validate the effectiveness of the proposed idea and the superiority of DTA over the state-of-the-art.



## **31. Reward Certification for Policy Smoothed Reinforcement Learning**

cs.LG

This paper will be presented in AAAI2024

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06436v2) [paper-pdf](http://arxiv.org/pdf/2312.06436v2)

**Authors**: Ronghui Mu, Leandro Soriano Marcolino, Tianle Zhang, Yanghao Zhang, Xiaowei Huang, Wenjie Ruan

**Abstract**: Reinforcement Learning (RL) has achieved remarkable success in safety-critical areas, but it can be weakened by adversarial attacks. Recent studies have introduced "smoothed policies" in order to enhance its robustness. Yet, it is still challenging to establish a provable guarantee to certify the bound of its total reward. Prior methods relied primarily on computing bounds using Lipschitz continuity or calculating the probability of cumulative reward above specific thresholds. However, these techniques are only suited for continuous perturbations on the RL agent's observations and are restricted to perturbations bounded by the $l_2$-norm. To address these limitations, this paper proposes a general black-box certification method capable of directly certifying the cumulative reward of the smoothed policy under various $l_p$-norm bounded perturbations. Furthermore, we extend our methodology to certify perturbations on action spaces. Our approach leverages f-divergence to measure the distinction between the original distribution and the perturbed distribution, subsequently determining the certification bound by solving a convex optimisation problem. We provide a comprehensive theoretical analysis and run sufficient experiments in multiple environments. Our results show that our method not only improves the certified lower bound of mean cumulative reward but also demonstrates better efficiency than state-of-the-art techniques.



## **32. Adversarial Driving: Attacking End-to-End Autonomous Driving**

cs.CV

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2103.09151v8) [paper-pdf](http://arxiv.org/pdf/2103.09151v8)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: As research in deep neural networks advances, deep convolutional networks become promising for autonomous driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for autonomous driving. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. While for regression tasks, the effect of adversarial attacks is not as well understood. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving models. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. In an average of 800 attacks with the same attack strength (epsilon=1), the image-specific and image-agnostic attack deviates the steering angle from the original output by 0.478 and 0.111, respectively, which is much stronger than random noises that only perturbs the steering angle by 0.002 (The steering angle ranges from [-1, 1]). Both attacks can be initiated in real-time on CPUs without employing GPUs. Demo video: https://youtu.be/I0i8uN2oOP0.



## **33. Adversarial Detection: Attacking Object Detection in Real Time**

cs.AI

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2209.01962v6) [paper-pdf](http://arxiv.org/pdf/2209.01962v6)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90% within about 20 iterations. The demo video is available at https://youtu.be/zJZ1aNlXsMU.



## **34. Cost Aware Untargeted Poisoning Attack against Graph Neural Networks,**

cs.AI

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07158v1) [paper-pdf](http://arxiv.org/pdf/2312.07158v1)

**Authors**: Yuwei Han, Yuni Lai, Yulin Zhu, Kai Zhou

**Abstract**: Graph Neural Networks (GNNs) have become widely used in the field of graph mining. However, these networks are vulnerable to structural perturbations. While many research efforts have focused on analyzing vulnerability through poisoning attacks, we have identified an inefficiency in current attack losses. These losses steer the attack strategy towards modifying edges targeting misclassified nodes or resilient nodes, resulting in a waste of structural adversarial perturbation. To address this issue, we propose a novel attack loss framework called the Cost Aware Poisoning Attack (CA-attack) to improve the allocation of the attack budget by dynamically considering the classification margins of nodes. Specifically, it prioritizes nodes with smaller positive margins while postponing nodes with negative margins. Our experiments demonstrate that the proposed CA-attack significantly enhances existing attack strategies



## **35. Data-Free Hard-Label Robustness Stealing Attack**

cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.05924v2) [paper-pdf](http://arxiv.org/pdf/2312.05924v2)

**Authors**: Xiaojian Yuan, Kejiang Chen, Wen Huang, Jie Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: The popularity of Machine Learning as a Service (MLaaS) has led to increased concerns about Model Stealing Attacks (MSA), which aim to craft a clone model by querying MLaaS. Currently, most research on MSA assumes that MLaaS can provide soft labels and that the attacker has a proxy dataset with a similar distribution. However, this fails to encapsulate the more practical scenario where only hard labels are returned by MLaaS and the data distribution remains elusive. Furthermore, most existing work focuses solely on stealing the model accuracy, neglecting the model robustness, while robustness is essential in security-sensitive scenarios, e.g., face-scan payment. Notably, improving model robustness often necessitates the use of expensive techniques such as adversarial training, thereby further making stealing robustness a more lucrative prospect. In response to these identified gaps, we introduce a novel Data-Free Hard-Label Robustness Stealing (DFHL-RS) attack in this paper, which enables the stealing of both model accuracy and robustness by simply querying hard labels of the target model without the help of any natural data. Comprehensive experiments demonstrate the effectiveness of our method. The clone model achieves a clean accuracy of 77.86% and a robust accuracy of 39.51% against AutoAttack, which are only 4.71% and 8.40% lower than the target model on the CIFAR-10 dataset, significantly exceeding the baselines. Our code is available at: https://github.com/LetheSec/DFHL-RS-Attack.



## **36. Divide-and-Conquer Attack: Harnessing the Power of LLM to Bypass the Censorship of Text-to-Image Generation Model**

cs.AI

20 pages,6 figures, under review

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07130v1) [paper-pdf](http://arxiv.org/pdf/2312.07130v1)

**Authors**: Yimo Deng, Huangxun Chen

**Abstract**: Text-to-image generative models offer many innovative services but also raise ethical concerns due to their potential to generate unethical images. Most publicly available text-to-image models employ safety filters to prevent unintended generation intents. In this work, we introduce the Divide-and-Conquer Attack to circumvent the safety filters of state-of-the-art text-to-image models. Our attack leverages LLMs as agents for text transformation, creating adversarial prompts from sensitive ones. We have developed effective helper prompts that enable LLMs to break down sensitive drawing prompts into multiple harmless descriptions, allowing them to bypass safety filters while still generating sensitive images. This means that the latent harmful meaning only becomes apparent when all individual elements are drawn together. Our evaluation demonstrates that our attack successfully circumvents the closed-box safety filter of SOTA DALLE-3 integrated natively into ChatGPT to generate unethical images. This approach, which essentially uses LLM-generated adversarial prompts against GPT-4-assisted DALLE-3, is akin to using one's own spear to breach their shield. It could have more severe security implications than previous manual crafting or iterative model querying methods, and we hope it stimulates more attention towards similar efforts. Our code and data are available at: https://github.com/researchcode001/Divide-and-Conquer-Attack



## **37. Promoting Counterfactual Robustness through Diversity**

cs.LG

Accepted at AAAI 2024

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06564v2) [paper-pdf](http://arxiv.org/pdf/2312.06564v2)

**Authors**: Francesco Leofante, Nico Potyka

**Abstract**: Counterfactual explanations shed light on the decisions of black-box models by explaining how an input can be altered to obtain a favourable decision from the model (e.g., when a loan application has been rejected). However, as noted recently, counterfactual explainers may lack robustness in the sense that a minor change in the input can cause a major change in the explanation. This can cause confusion on the user side and open the door for adversarial attacks. In this paper, we study some sources of non-robustness. While there are fundamental reasons for why an explainer that returns a single counterfactual cannot be robust in all instances, we show that some interesting robustness guarantees can be given by reporting multiple rather than a single counterfactual. Unfortunately, the number of counterfactuals that need to be reported for the theoretical guarantees to hold can be prohibitively large. We therefore propose an approximation algorithm that uses a diversity criterion to select a feasible number of most relevant explanations and study its robustness empirically. Our experiments indicate that our method improves the state-of-the-art in generating robust explanations, while maintaining other desirable properties and providing competitive computational performance.



## **38. Patch-MI: Enhancing Model Inversion Attacks via Patch-Based Reconstruction**

cs.AI

11 pages

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07040v1) [paper-pdf](http://arxiv.org/pdf/2312.07040v1)

**Authors**: Jonggyu Jang, Hyeonsu Lyu, Hyun Jong Yang

**Abstract**: Model inversion (MI) attacks aim to reveal sensitive information in training datasets by solely accessing model weights. Generative MI attacks, a prominent strand in this field, utilize auxiliary datasets to recreate target data attributes, restricting the images to remain photo-realistic, but their success often depends on the similarity between auxiliary and target datasets. If the distributions are dissimilar, existing MI attack attempts frequently fail, yielding unrealistic or target-unrelated results. In response to these challenges, we introduce a groundbreaking approach named Patch-MI, inspired by jigsaw puzzle assembly. To this end, we build upon a new probabilistic interpretation of MI attacks, employing a generative adversarial network (GAN)-like framework with a patch-based discriminator. This approach allows the synthesis of images that are similar to the target dataset distribution, even in cases of dissimilar auxiliary dataset distribution. Moreover, we artfully employ a random transformation block, a sophisticated maneuver that crafts generalized images, thus enhancing the efficacy of the target classifier. Our numerical and graphical findings demonstrate that Patch-MI surpasses existing generative MI methods in terms of accuracy, marking significant advancements while preserving comparable statistical dataset quality. For reproducibility of our results, we make our source code publicly available in https://github.com/jonggyujang0123/Patch-Attack.



## **39. EdgePruner: Poisoned Edge Pruning in Graph Contrastive Learning**

cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07022v1) [paper-pdf](http://arxiv.org/pdf/2312.07022v1)

**Authors**: Hiroya Kato, Kento Hasegawa, Seira Hidano, Kazuhide Fukushima

**Abstract**: Graph Contrastive Learning (GCL) is unsupervised graph representation learning that can obtain useful representation of unknown nodes. The node representation can be utilized as features of downstream tasks. However, GCL is vulnerable to poisoning attacks as with existing learning models. A state-of-the-art defense cannot sufficiently negate adverse effects by poisoned graphs although such a defense introduces adversarial training in the GCL. To achieve further improvement, pruning adversarial edges is important. To the best of our knowledge, the feasibility remains unexplored in the GCL domain. In this paper, we propose a simple defense for GCL, EdgePruner. We focus on the fact that the state-of-the-art poisoning attack on GCL tends to mainly add adversarial edges to create poisoned graphs, which means that pruning edges is important to sanitize the graphs. Thus, EdgePruner prunes edges that contribute to minimizing the contrastive loss based on the node representation obtained after training on poisoned graphs by GCL. Furthermore, we focus on the fact that nodes with distinct features are connected by adversarial edges in poisoned graphs. Thus, we introduce feature similarity between neighboring nodes to help more appropriately determine adversarial edges. This similarity is helpful in further eliminating adverse effects from poisoned graphs on various datasets. Finally, EdgePruner outputs a graph that yields the minimum contrastive loss as the sanitized graph. Our results demonstrate that pruning adversarial edges is feasible on six datasets. EdgePruner can improve the accuracy of node classification under the attack by up to 5.55% compared with that of the state-of-the-art defense. Moreover, we show that EdgePruner is immune to an adaptive attack.



## **40. Attacking the Loop: Adversarial Attacks on Graph-based Loop Closure Detection**

cs.CV

Accepted at VISIGRAPP 2024, 8 pages

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06991v1) [paper-pdf](http://arxiv.org/pdf/2312.06991v1)

**Authors**: Jonathan J. Y. Kim, Martin Urschler, Patricia J. Riddle, Jorg S. Wicker

**Abstract**: With the advancement in robotics, it is becoming increasingly common for large factories and warehouses to incorporate visual SLAM (vSLAM) enabled automated robots that operate closely next to humans. This makes any adversarial attacks on vSLAM components potentially detrimental to humans working alongside them. Loop Closure Detection (LCD) is a crucial component in vSLAM that minimizes the accumulation of drift in mapping, since even a small drift can accumulate into a significant drift over time. A prior work by Kim et al., SymbioLCD2, unified visual features and semantic objects into a single graph structure for finding loop closure candidates. While this provided a performance improvement over visual feature-based LCD, it also created a single point of vulnerability for potential graph-based adversarial attacks. Unlike previously reported visual-patch based attacks, small graph perturbations are far more challenging to detect, making them a more significant threat. In this paper, we present Adversarial-LCD, a novel black-box evasion attack framework that employs an eigencentrality-based perturbation method and an SVM-RBF surrogate model with a Weisfeiler-Lehman feature extractor for attacking graph-based LCD. Our evaluation shows that the attack performance of Adversarial-LCD with the SVM-RBF surrogate model was superior to that of other machine learning surrogate algorithms, including SVM-linear, SVM-polynomial, and Bayesian classifier, demonstrating the effectiveness of our attack framework. Furthermore, we show that our eigencentrality-based perturbation method outperforms other algorithms, such as Random-walk and Shortest-path, highlighting the efficiency of Adversarial-LCD's perturbation selection method.



## **41. Task-Agnostic Privacy-Preserving Representation Learning for Federated Learning Against Attribute Inference Attacks**

cs.CR

Accepted by AAAI 2024; Full version

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06989v1) [paper-pdf](http://arxiv.org/pdf/2312.06989v1)

**Authors**: Caridad Arroyo Arevalo, Sayedeh Leila Noorbakhsh, Yun Dong, Yuan Hong, Binghui Wang

**Abstract**: Federated learning (FL) has been widely studied recently due to its property to collaboratively train data from different devices without sharing the raw data. Nevertheless, recent studies show that an adversary can still be possible to infer private information about devices' data, e.g., sensitive attributes such as income, race, and sexual orientation. To mitigate the attribute inference attacks, various existing privacy-preserving FL methods can be adopted/adapted. However, all these existing methods have key limitations: they need to know the FL task in advance, or have intolerable computational overheads or utility losses, or do not have provable privacy guarantees.   We address these issues and design a task-agnostic privacy-preserving presentation learning method for FL ({\bf TAPPFL}) against attribute inference attacks. TAPPFL is formulated via information theory. Specifically, TAPPFL has two mutual information goals, where one goal learns task-agnostic data representations that contain the least information about the private attribute in each device's data, and the other goal ensures the learnt data representations include as much information as possible about the device data to maintain FL utility. We also derive privacy guarantees of TAPPFL against worst-case attribute inference attacks, as well as the inherent tradeoff between utility preservation and privacy protection. Extensive results on multiple datasets and applications validate the effectiveness of TAPPFL to protect data privacy, maintain the FL utility, and be efficient as well. Experimental results also show that TAPPFL outperforms the existing defenses\footnote{Source code and full version: \url{https://github.com/TAPPFL}}.



## **42. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2311.06062v2) [paper-pdf](http://arxiv.org/pdf/2311.06062v2)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.



## **43. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

cs.CL

17 pages,10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06924v1) [paper-pdf](http://arxiv.org/pdf/2312.06924v1)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integraty of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.



## **44. Adversarial Estimation of Topological Dimension with Harmonic Score Maps**

cs.LG

Accepted to the NeurIPS'23 Workshop on Diffusion Models

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06869v1) [paper-pdf](http://arxiv.org/pdf/2312.06869v1)

**Authors**: Eric Yeats, Cameron Darwin, Frank Liu, Hai Li

**Abstract**: Quantification of the number of variables needed to locally explain complex data is often the first step to better understanding it. Existing techniques from intrinsic dimension estimation leverage statistical models to glean this information from samples within a neighborhood. However, existing methods often rely on well-picked hyperparameters and ample data as manifold dimension and curvature increases. Leveraging insight into the fixed point of the score matching objective as the score map is regularized by its Dirichlet energy, we show that it is possible to retrieve the topological dimension of the manifold learned by the score map. We then introduce a novel method to measure the learned manifold's topological dimension (i.e., local intrinsic dimension) using adversarial attacks, thereby generating useful interpretations of the learned manifold.



## **45. Adversarial Purification with the Manifold Hypothesis**

cs.LG

Extended version of paper accepted at AAAI 2024 with supplementary  materials

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2210.14404v4) [paper-pdf](http://arxiv.org/pdf/2210.14404v4)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework for adversarial robustness using the manifold hypothesis. This framework provides sufficient conditions for defending against adversarial examples. We develop an adversarial purification method with this framework. Our method combines manifold learning with variational inference to provide adversarial robustness without the need for expensive adversarial training. Experimentally, our approach can provide adversarial robustness even if attackers are aware of the existence of the defense. In addition, our method can also serve as a test-time defense mechanism for variational autoencoders.



## **46. Sparse but Strong: Crafting Adversarially Robust Graph Lottery Tickets**

cs.LG

Accepted at NeurIPS 2023 GLFrontiers Workshop

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06568v1) [paper-pdf](http://arxiv.org/pdf/2312.06568v1)

**Authors**: Subhajit Dutta Chowdhury, Zhiyu Ni, Qingyuan Peng, Souvik Kundu, Pierluigi Nuzzo

**Abstract**: Graph Lottery Tickets (GLTs), comprising a sparse adjacency matrix and a sparse graph neural network (GNN), can significantly reduce the inference latency and compute footprint compared to their dense counterparts. Despite these benefits, their performance against adversarial structure perturbations remains to be fully explored. In this work, we first investigate the resilience of GLTs against different structure perturbation attacks and observe that they are highly vulnerable and show a large drop in classification accuracy. Based on this observation, we then present an adversarially robust graph sparsification (ARGS) framework that prunes the adjacency matrix and the GNN weights by optimizing a novel loss function capturing the graph homophily property and information associated with both the true labels of the train nodes and the pseudo labels of the test nodes. By iteratively applying ARGS to prune both the perturbed graph adjacency matrix and the GNN model weights, we can find adversarially robust graph lottery tickets that are highly sparse yet achieve competitive performance under different untargeted training-time structure attacks. Evaluations conducted on various benchmarks, considering different poisoning structure attacks, namely, PGD, MetaAttack, Meta-PGD, and PR-BCD demonstrate that the GLTs generated by ARGS can significantly improve the robustness, even when subjected to high levels of sparsity.



## **47. Robust Graph Neural Network based on Graph Denoising**

cs.LG

Presented in the 2023 Asilomar Conference on Signals, Systems, and  Computers (Oct. 29th - Nov 1st, 2023)

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06557v1) [paper-pdf](http://arxiv.org/pdf/2312.06557v1)

**Authors**: Victor M. Tenorio, Samuel Rey, Antonio G. Marques

**Abstract**: Graph Neural Networks (GNNs) have emerged as a notorious alternative to address learning problems dealing with non-Euclidean datasets. However, although most works assume that the graph is perfectly known, the observed topology is prone to errors stemming from observational noise, graph-learning limitations, or adversarial attacks. If ignored, these perturbations may drastically hinder the performance of GNNs. To address this limitation, this work proposes a robust implementation of GNNs that explicitly accounts for the presence of perturbations in the observed topology. For any task involving GNNs, our core idea is to i) solve an optimization problem not only over the learnable parameters of the GNN but also over the true graph, and ii) augment the fitting cost with a term accounting for discrepancies on the graph. Specifically, we consider a convolutional GNN based on graph filters and follow an alternating optimization approach to handle the (non-differentiable and constrained) optimization problem by combining gradient descent and projected proximal updates. The resulting algorithm is not limited to a particular type of graph and is amenable to incorporating prior information about the perturbations. Finally, we assess the performance of the proposed method through several numerical experiments.



## **48. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks**

cs.CV

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2306.09124v3) [paper-pdf](http://arxiv.org/pdf/2306.09124v3)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications, yet current research in this area is unsatisfactory. In this paper, we propose DIFFender, a novel defense method that leverages a text-guided diffusion model to defend against adversarial patches. DIFFender includes two main stages: patch localization and patch restoration. In the localization stage, we find and exploit an intriguing property of the diffusion model to precisely identify the locations of adversarial patches. In the restoration stage, we employ the diffusion model to reconstruct the adversarial regions in the images while preserving the integrity of the visual content. Thanks to the former finding, these two stages can be simultaneously guided by a unified diffusion model. Thus, we can utilize the close interaction between them to improve the whole defense performance. Moreover, we propose a few-shot prompt-tuning algorithm to fine-tune the diffusion model, enabling the pre-trained diffusion model to adapt to the defense task easily. We conduct extensive experiments on image classification, face recognition, and further in the physical world, demonstrating that our proposed method exhibits superior robustness under strong adaptive attacks and generalizes well across various scenarios, diverse classifiers, and multiple patch attack methods.



## **49. MalPurifier: Enhancing Android Malware Detection with Adversarial Purification against Evasion Attacks**

cs.CR

14 pages; In submission

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06423v1) [paper-pdf](http://arxiv.org/pdf/2312.06423v1)

**Authors**: Yuyang Zhou, Guang Cheng, Zongyao Chen, Shui Yu

**Abstract**: Machine learning (ML) has gained significant adoption in Android malware detection to address the escalating threats posed by the rapid proliferation of malware attacks. However, recent studies have revealed the inherent vulnerabilities of ML-based detection systems to evasion attacks. While efforts have been made to address this critical issue, many of the existing defensive methods encounter challenges such as lower effectiveness or reduced generalization capabilities. In this paper, we introduce a novel Android malware detection method, MalPurifier, which exploits adversarial purification to eliminate perturbations independently, resulting in attack mitigation in a light and flexible way. Specifically, MalPurifier employs a Denoising AutoEncoder (DAE)-based purification model to preprocess input samples, removing potential perturbations from them and then leading to correct classification. To enhance defense effectiveness, we propose a diversified adversarial perturbation mechanism that strengthens the purification model against different manipulations from various evasion attacks. We also incorporate randomized "protective noises" onto benign samples to prevent excessive purification. Furthermore, we customize a loss function for improving the DAE model, combining reconstruction loss and prediction loss, to enhance feature representation learning, resulting in accurate reconstruction and classification. Experimental results on two Android malware datasets demonstrate that MalPurifier outperforms the state-of-the-art defenses, and it significantly strengthens the vulnerable malware detector against 37 evasion attacks, achieving accuracies over 90.91%. Notably, MalPurifier demonstrates easy scalability to other detectors, offering flexibility and robustness in its implementation.



## **50. Bidirectional Contrastive Split Learning for Visual Question Answering**

cs.CV

Accepted for AAAI 2024

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2208.11435v4) [paper-pdf](http://arxiv.org/pdf/2208.11435v4)

**Authors**: Yuwei Sun, Hideya Ochiai

**Abstract**: Visual Question Answering (VQA) based on multi-modal data facilitates real-life applications such as home robots and medical diagnoses. One significant challenge is to devise a robust decentralized learning framework for various client models where centralized data collection is refrained due to confidentiality concerns. This work aims to tackle privacy-preserving VQA by decoupling a multi-modal model into representation modules and a contrastive module and leveraging inter-module gradients sharing and inter-client weight sharing. To this end, we propose Bidirectional Contrastive Split Learning (BiCSL) to train a global multi-modal model on the entire data distribution of decentralized clients. We employ the contrastive loss that enables a more efficient self-supervised learning of decentralized modules. Comprehensive experiments are conducted on the VQA-v2 dataset based on five SOTA VQA models, demonstrating the effectiveness of the proposed method. Furthermore, we inspect BiCSL's robustness against a dual-key backdoor attack on VQA. Consequently, BiCSL shows much better robustness to the multi-modal adversarial attack compared to the centralized learning method, which provides a promising approach to decentralized multi-modal learning.



