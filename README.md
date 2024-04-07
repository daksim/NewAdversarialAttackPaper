# Latest Adversarial Attack Papers
**update at 2024-04-07 11:06:26**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Knowledge Distillation-Based Model Extraction Attack using Private Counterfactual Explanations**

cs.LG

15 pages

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03348v1) [paper-pdf](http://arxiv.org/pdf/2404.03348v1)

**Authors**: Fatima Ezzeddine, Omran Ayoub, Silvia Giordano

**Abstract**: In recent years, there has been a notable increase in the deployment of machine learning (ML) models as services (MLaaS) across diverse production software applications. In parallel, explainable AI (XAI) continues to evolve, addressing the necessity for transparency and trustworthiness in ML models. XAI techniques aim to enhance the transparency of ML models by providing insights, in terms of the model's explanations, into their decision-making process. Simultaneously, some MLaaS platforms now offer explanations alongside the ML prediction outputs. This setup has elevated concerns regarding vulnerabilities in MLaaS, particularly in relation to privacy leakage attacks such as model extraction attacks (MEA). This is due to the fact that explanations can unveil insights about the inner workings of the model which could be exploited by malicious users. In this work, we focus on investigating how model explanations, particularly Generative adversarial networks (GANs)-based counterfactual explanations (CFs), can be exploited for performing MEA within the MLaaS platform. We also delve into assessing the effectiveness of incorporating differential privacy (DP) as a mitigation strategy. To this end, we first propose a novel MEA methodology based on Knowledge Distillation (KD) to enhance the efficiency of extracting a substitute model of a target model exploiting CFs. Then, we advise an approach for training CF generators incorporating DP to generate private CFs. We conduct thorough experimental evaluations on real-world datasets and demonstrate that our proposed KD-based MEA can yield a high-fidelity substitute model with reduced queries with respect to baseline approaches. Furthermore, our findings reveal that the inclusion of a privacy layer impacts the performance of the explainer, the quality of CFs, and results in a reduction in the MEA performance.



## **2. Meta Invariance Defense Towards Generalizable Robustness to Unknown Adversarial Attacks**

cs.CV

Accepted by IEEE TPAMI in 2024

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03340v1) [paper-pdf](http://arxiv.org/pdf/2404.03340v1)

**Authors**: Lei Zhang, Yuhang Zhou, Yi Yang, Xinbo Gao

**Abstract**: Despite providing high-performance solutions for computer vision tasks, the deep neural network (DNN) model has been proved to be extremely vulnerable to adversarial attacks. Current defense mainly focuses on the known attacks, but the adversarial robustness to the unknown attacks is seriously overlooked. Besides, commonly used adaptive learning and fine-tuning technique is unsuitable for adversarial defense since it is essentially a zero-shot problem when deployed. Thus, to tackle this challenge, we propose an attack-agnostic defense method named Meta Invariance Defense (MID). Specifically, various combinations of adversarial attacks are randomly sampled from a manually constructed Attacker Pool to constitute different defense tasks against unknown attacks, in which a student encoder is supervised by multi-consistency distillation to learn the attack-invariant features via a meta principle. The proposed MID has two merits: 1) Full distillation from pixel-, feature- and prediction-level between benign and adversarial samples facilitates the discovery of attack-invariance. 2) The model simultaneously achieves robustness to the imperceptible adversarial perturbations in high-level image classification and attack-suppression in low-level robust image regeneration. Theoretical and empirical studies on numerous benchmarks such as ImageNet verify the generalizable robustness and superiority of MID under various attacks.



## **3. Learn What You Want to Unlearn: Unlearning Inversion Attacks against Machine Unlearning**

cs.CR

To Appear in the 45th IEEE Symposium on Security and Privacy, May  20-23, 2024

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03233v1) [paper-pdf](http://arxiv.org/pdf/2404.03233v1)

**Authors**: Hongsheng Hu, Shuo Wang, Tian Dong, Minhui Xue

**Abstract**: Machine unlearning has become a promising solution for fulfilling the "right to be forgotten", under which individuals can request the deletion of their data from machine learning models. However, existing studies of machine unlearning mainly focus on the efficacy and efficiency of unlearning methods, while neglecting the investigation of the privacy vulnerability during the unlearning process. With two versions of a model available to an adversary, that is, the original model and the unlearned model, machine unlearning opens up a new attack surface. In this paper, we conduct the first investigation to understand the extent to which machine unlearning can leak the confidential content of the unlearned data. Specifically, under the Machine Learning as a Service setting, we propose unlearning inversion attacks that can reveal the feature and label information of an unlearned sample by only accessing the original and unlearned model. The effectiveness of the proposed unlearning inversion attacks is evaluated through extensive experiments on benchmark datasets across various model architectures and on both exact and approximate representative unlearning approaches. The experimental results indicate that the proposed attack can reveal the sensitive information of the unlearned data. As such, we identify three possible defenses that help to mitigate the proposed attacks, while at the cost of reducing the utility of the unlearned model. The study in this paper uncovers an underexplored gap between machine unlearning and the privacy of unlearned data, highlighting the need for the careful design of mechanisms for implementing unlearning without leaking the information of the unlearned data.



## **4. FACTUAL: A Novel Framework for Contrastive Learning Based Robust SAR Image Classification**

cs.CV

2024 IEEE Radar Conference

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03225v1) [paper-pdf](http://arxiv.org/pdf/2404.03225v1)

**Authors**: Xu Wang, Tian Ye, Rajgopal Kannan, Viktor Prasanna

**Abstract**: Deep Learning (DL) Models for Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR), while delivering improved performance, have been shown to be quite vulnerable to adversarial attacks. Existing works improve robustness by training models on adversarial samples. However, by focusing mostly on attacks that manipulate images randomly, they neglect the real-world feasibility of such attacks. In this paper, we propose FACTUAL, a novel Contrastive Learning framework for Adversarial Training and robust SAR classification. FACTUAL consists of two components: (1) Differing from existing works, a novel perturbation scheme that incorporates realistic physical adversarial attacks (such as OTSA) to build a supervised adversarial pre-training network. This network utilizes class labels for clustering clean and perturbed images together into a more informative feature space. (2) A linear classifier cascaded after the encoder to use the computed representations to predict the target labels. By pre-training and fine-tuning our model on both clean and adversarial samples, we show that our model achieves high prediction accuracy on both cases. Our model achieves 99.7% accuracy on clean samples, and 89.6% on perturbed samples, both outperforming previous state-of-the-art methods.



## **5. Robust Federated Learning for Wireless Networks: A Demonstration with Channel Estimation**

cs.LG

Submitted to IEEE GLOBECOM 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.03088v1) [paper-pdf](http://arxiv.org/pdf/2404.03088v1)

**Authors**: Zexin Fang, Bin Han, Hans D. Schotten

**Abstract**: Federated learning (FL) offers a privacy-preserving collaborative approach for training models in wireless networks, with channel estimation emerging as a promising application. Despite extensive studies on FL-empowered channel estimation, the security concerns associated with FL require meticulous attention. In a scenario where small base stations (SBSs) serve as local models trained on cached data, and a macro base station (MBS) functions as the global model setting, an attacker can exploit the vulnerability of FL, launching attacks with various adversarial attacks or deployment tactics. In this paper, we analyze such vulnerabilities, corresponding solutions were brought forth, and validated through simulation.



## **6. Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning**

cs.CR

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2306.05494v2) [paper-pdf](http://arxiv.org/pdf/2306.05494v2)

**Authors**: Mohamed el Shehaby, Ashraf Matrawy

**Abstract**: Machine Learning (ML) has become ubiquitous, and its deployment in Network Intrusion Detection Systems (NIDS) is inevitable due to its automated nature and high accuracy compared to traditional models in processing and classifying large volumes of data. However, ML has been found to have several flaws, most importantly, adversarial attacks, which aim to trick ML models into producing faulty predictions. While most adversarial attack research focuses on computer vision datasets, recent studies have explored the suitability of these attacks against ML-based network security entities, especially NIDS, due to the wide difference between different domains regarding the generation of adversarial attacks.   To further explore the practicality of adversarial attacks against ML-based NIDS in-depth, this paper presents three distinct contributions: identifying numerous practicality issues for evasion adversarial attacks on ML-NIDS using an attack tree threat model, introducing a taxonomy of practicality issues associated with adversarial attacks against ML-based NIDS, and investigating how the dynamicity of some real-world ML models affects adversarial attacks against NIDS. Our experiments indicate that continuous re-training, even without adversarial training, can reduce the effectiveness of adversarial attacks. While adversarial attacks can compromise ML-based NIDSs, our aim is to highlight the significant gap between research and real-world practicality in this domain, warranting attention.



## **7. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.03027v1) [paper-pdf](http://arxiv.org/pdf/2404.03027v1)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.



## **8. Adversarial Attacks and Dimensionality in Text Classifiers**

cs.LG

This paper is accepted for publication at EURASIP Journal on  Information Security in 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02660v1) [paper-pdf](http://arxiv.org/pdf/2404.02660v1)

**Authors**: Nandish Chattopadhyay, Atreya Goswami, Anupam Chattopadhyay

**Abstract**: Adversarial attacks on machine learning algorithms have been a key deterrent to the adoption of AI in many real-world use cases. They significantly undermine the ability of high-performance neural networks by forcing misclassifications. These attacks introduce minute and structured perturbations or alterations in the test samples, imperceptible to human annotators in general, but trained neural networks and other models are sensitive to it. Historically, adversarial attacks have been first identified and studied in the domain of image processing. In this paper, we study adversarial examples in the field of natural language processing, specifically text classification tasks. We investigate the reasons for adversarial vulnerability, particularly in relation to the inherent dimensionality of the model. Our key finding is that there is a very strong correlation between the embedding dimensionality of the adversarial samples and their effectiveness on models tuned with input samples with same embedding dimension. We utilize this sensitivity to design an adversarial defense mechanism. We use ensemble models of varying inherent dimensionality to thwart the attacks. This is tested on multiple datasets for its efficacy in providing robustness. We also study the problem of measuring adversarial perturbation using different distance metrics. For all of the aforementioned studies, we have run tests on multiple models with varying dimensionality and used a word-vector level adversarial attack to substantiate the findings.



## **9. Adversary-Augmented Simulation to evaluate fairness on HyperLedger Fabric**

cs.CR

10 pages, 8 figures

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2403.14342v2) [paper-pdf](http://arxiv.org/pdf/2403.14342v2)

**Authors**: Erwan Mahe, Rouwaida Abdallah, Sara Tucci-Piergiovanni, Pierre-Yves Piriou

**Abstract**: This paper presents a novel adversary model specifically tailored to distributed systems, aiming to assess the security of blockchain networks. Building upon concepts such as adversarial assumptions, goals, and capabilities, our proposed adversary model classifies and constrains the use of adversarial actions based on classical distributed system models, defined by both failure and communication models. The objective is to study the effects of these allowed actions on the properties of distributed protocols under various system models. A significant aspect of our research involves integrating this adversary model into the Multi-Agent eXperimenter (MAX) framework. This integration enables fine-grained simulations of adversarial attacks on blockchain networks. In this paper, we particularly study four distinct fairness properties on Hyperledger Fabric with the Byzantine Fault Tolerant Tendermint consensus algorithm being selected for its ordering service. We define novel attacks that combine adversarial actions on both protocols, with the aim of violating a specific client-fairness property. Simulations confirm our ability to violate this property and allow us to evaluate the impact of these attacks on several order-fairness properties that relate orders of transaction reception and delivery.



## **10. Unsegment Anything by Simulating Deformation**

cs.CV

CVPR 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02585v1) [paper-pdf](http://arxiv.org/pdf/2404.02585v1)

**Authors**: Jiahao Lu, Xingyi Yang, Xinchao Wang

**Abstract**: Foundation segmentation models, while powerful, pose a significant risk: they enable users to effortlessly extract any objects from any digital content with a single click, potentially leading to copyright infringement or malicious misuse. To mitigate this risk, we introduce a new task "Anything Unsegmentable" to grant any image "the right to be unsegmented". The ambitious pursuit of the task is to achieve highly transferable adversarial attacks against all prompt-based segmentation models, regardless of model parameterizations and prompts. We highlight the non-transferable and heterogeneous nature of prompt-specific adversarial noises. Our approach focuses on disrupting image encoder features to achieve prompt-agnostic attacks. Intriguingly, targeted feature attacks exhibit better transferability compared to untargeted ones, suggesting the optimal update direction aligns with the image manifold. Based on the observations, we design a novel attack named Unsegment Anything by Simulating Deformation (UAD). Our attack optimizes a differentiable deformation function to create a target deformed image, which alters structural information while preserving achievable feature distance by adversarial example. Extensive experiments verify the effectiveness of our approach, compromising a variety of promptable segmentation models with different architectures and prompt interfaces. We release the code at https://github.com/jiahaolu97/anything-unsegmentable.



## **11. A Unified Membership Inference Method for Visual Self-supervised Encoder via Part-aware Capability**

cs.CV

Membership Inference, Self-supervised learning

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02462v1) [paper-pdf](http://arxiv.org/pdf/2404.02462v1)

**Authors**: Jie Zhu, Jirong Zha, Ding Li, Leye Wang

**Abstract**: Self-supervised learning shows promise in harnessing extensive unlabeled data, but it also confronts significant privacy concerns, especially in vision. In this paper, we aim to perform membership inference on visual self-supervised models in a more realistic setting: self-supervised training method and details are unknown for an adversary when attacking as he usually faces a black-box system in practice. In this setting, considering that self-supervised model could be trained by completely different self-supervised paradigms, e.g., masked image modeling and contrastive learning, with complex training details, we propose a unified membership inference method called PartCrop. It is motivated by the shared part-aware capability among models and stronger part response on the training data. Specifically, PartCrop crops parts of objects in an image to query responses with the image in representation space. We conduct extensive attacks on self-supervised models with different training protocols and structures using three widely used image datasets. The results verify the effectiveness and generalization of PartCrop. Moreover, to defend against PartCrop, we evaluate two common approaches, i.e., early stop and differential privacy, and propose a tailored method called shrinking crop scale range. The defense experiments indicate that all of them are effective. Our code is available at https://github.com/JiePKU/PartCrop



## **12. Designing a Photonic Physically Unclonable Function Having Resilience to Machine Learning Attacks**

cs.CR

14 pages, 8 figures

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02440v1) [paper-pdf](http://arxiv.org/pdf/2404.02440v1)

**Authors**: Elena R. Henderson, Jessie M. Henderson, Hiva Shahoei, William V. Oxford, Eric C. Larson, Duncan L. MacFarlane, Mitchell A. Thornton

**Abstract**: Physically unclonable functions (PUFs) are designed to act as device 'fingerprints.' Given an input challenge, the PUF circuit should produce an unpredictable response for use in situations such as root-of-trust applications and other hardware-level cybersecurity applications. PUFs are typically subcircuits present within integrated circuits (ICs), and while conventional IC PUFs are well-understood, several implementations have proven vulnerable to malicious exploits, including those perpetrated by machine learning (ML)-based attacks. Such attacks can be difficult to prevent because they are often designed to work even when relatively few challenge-response pairs are known in advance. Hence the need for both more resilient PUF designs and analysis of ML-attack susceptibility. Previous work has developed a PUF for photonic integrated circuits (PICs). A PIC PUF not only produces unpredictable responses given manufacturing-introduced tolerances, but is also less prone to electromagnetic radiation eavesdropping attacks than a purely electronic IC PUF. In this work, we analyze the resilience of the proposed photonic PUF when subjected to ML-based attacks. Specifically, we describe a computational PUF model for producing the large datasets required for training ML attacks; we analyze the quality of the model; and we discuss the modeled PUF's susceptibility to ML-based attacks. We find that the modeled PUF generates distributions that resemble uniform white noise, explaining the exhibited resilience to neural-network-based attacks designed to exploit latent relationships between challenges and responses. Preliminary analysis suggests that the PUF exhibits similar resilience to generative adversarial networks, and continued development will show whether more-sophisticated ML approaches better compromise the PUF and -- if so -- how design modifications might improve resilience.



## **13. One Noise to Rule Them All: Multi-View Adversarial Attacks with Universal Perturbation**

cs.CV

6 pages, 4 figures, presented at ICAIA, Springer to publish under  Algorithms for Intelligent Systems

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02287v1) [paper-pdf](http://arxiv.org/pdf/2404.02287v1)

**Authors**: Mehmet Ergezer, Phat Duong, Christian Green, Tommy Nguyen, Abdurrahman Zeybey

**Abstract**: This paper presents a novel universal perturbation method for generating robust multi-view adversarial examples in 3D object recognition. Unlike conventional attacks limited to single views, our approach operates on multiple 2D images, offering a practical and scalable solution for enhancing model scalability and robustness. This generalizable method bridges the gap between 2D perturbations and 3D-like attack capabilities, making it suitable for real-world applications.   Existing adversarial attacks may become ineffective when images undergo transformations like changes in lighting, camera position, or natural deformations. We address this challenge by crafting a single universal noise perturbation applicable to various object views. Experiments on diverse rendered 3D objects demonstrate the effectiveness of our approach. The universal perturbation successfully identified a single adversarial noise for each given set of 3D object renders from multiple poses and viewpoints. Compared to single-view attacks, our universal attacks lower classification confidence across multiple viewing angles, especially at low noise levels. A sample implementation is made available at https://github.com/memoatwit/UniversalPerturbation.



## **14. Towards Robust 3D Pose Transfer with Adversarial Learning**

cs.CV

CVPR 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02242v1) [paper-pdf](http://arxiv.org/pdf/2404.02242v1)

**Authors**: Haoyu Chen, Hao Tang, Ehsan Adeli, Guoying Zhao

**Abstract**: 3D pose transfer that aims to transfer the desired pose to a target mesh is one of the most challenging 3D generation tasks. Previous attempts rely on well-defined parametric human models or skeletal joints as driving pose sources. However, to obtain those clean pose sources, cumbersome but necessary pre-processing pipelines are inevitable, hindering implementations of the real-time applications. This work is driven by the intuition that the robustness of the model can be enhanced by introducing adversarial samples into the training, leading to a more invulnerable model to the noisy inputs, which even can be further extended to directly handling the real-world data like raw point clouds/scans without intermediate processing. Furthermore, we propose a novel 3D pose Masked Autoencoder (3D-PoseMAE), a customized MAE that effectively learns 3D extrinsic presentations (i.e., pose). 3D-PoseMAE facilitates learning from the aspect of extrinsic attributes by simultaneously generating adversarial samples that perturb the model and learning the arbitrary raw noisy poses via a multi-scale masking strategy. Both qualitative and quantitative studies show that the transferred meshes given by our network result in much better quality. Besides, we demonstrate the strong generalizability of our method on various poses, different domains, and even raw scans. Experimental results also show meaningful insights that the intermediate adversarial samples generated in the training can successfully attack the existing pose transfer models.



## **15. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02151v1) [paper-pdf](http://arxiv.org/pdf/2404.02151v1)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). We provide the code, prompts, and logs of the attacks at https://github.com/tml-epfl/llm-adaptive-attacks.



## **16. READ: Improving Relation Extraction from an ADversarial Perspective**

cs.CL

Accepted by findings of NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02931v1) [paper-pdf](http://arxiv.org/pdf/2404.02931v1)

**Authors**: Dawei Li, William Hogan, Jingbo Shang

**Abstract**: Recent works in relation extraction (RE) have achieved promising benchmark accuracy; however, our adversarial attack experiments show that these works excessively rely on entities, making their generalization capability questionable. To address this issue, we propose an adversarial training method specifically designed for RE. Our approach introduces both sequence- and token-level perturbations to the sample and uses a separate perturbation vocabulary to improve the search for entity and context perturbations. Furthermore, we introduce a probabilistic strategy for leaving clean tokens in the context during adversarial training. This strategy enables a larger attack budget for entities and coaxes the model to leverage relational patterns embedded in the context. Extensive experiments show that compared to various adversarial training methods, our method significantly improves both the accuracy and robustness of the model. Additionally, experiments on different data availability settings highlight the effectiveness of our method in low-resource scenarios. We also perform in-depth analyses of our proposed method and provide further hints. We will release our code at https://github.com/David-Li0406/READ.



## **17. Red-Teaming Segment Anything Model**

cs.CV

CVPR 2024 - The 4th Workshop of Adversarial Machine Learning on  Computer Vision: Robustness of Foundation Models

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02067v1) [paper-pdf](http://arxiv.org/pdf/2404.02067v1)

**Authors**: Krzysztof Jankowski, Bartlomiej Sobieski, Mateusz Kwiatkowski, Jakub Szulc, Michal Janik, Hubert Baniecki, Przemyslaw Biecek

**Abstract**: Foundation models have emerged as pivotal tools, tackling many complex tasks through pre-training on vast datasets and subsequent fine-tuning for specific applications. The Segment Anything Model is one of the first and most well-known foundation models for computer vision segmentation tasks. This work presents a multi-faceted red-teaming analysis that tests the Segment Anything Model against challenging tasks: (1) We analyze the impact of style transfer on segmentation masks, demonstrating that applying adverse weather conditions and raindrops to dashboard images of city roads significantly distorts generated masks. (2) We focus on assessing whether the model can be used for attacks on privacy, such as recognizing celebrities' faces, and show that the model possesses some undesired knowledge in this task. (3) Finally, we check how robust the model is to adversarial attacks on segmentation masks under text prompts. We not only show the effectiveness of popular white-box attacks and resistance to black-box attacks but also introduce a novel approach - Focused Iterative Gradient Attack (FIGA) that combines white-box approaches to construct an efficient attack resulting in a smaller number of modified pixels. All of our testing methods and analyses indicate a need for enhanced safety measures in foundation models for image segmentation.



## **18. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

cs.CL

NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2311.09447v2) [paper-pdf](http://arxiv.org/pdf/2311.09447v2)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an adversarial assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose advCoU, an extended Chain of Utterances-based (CoU) prompting strategy by incorporating carefully crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.



## **19. Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy**

cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.16591v3) [paper-pdf](http://arxiv.org/pdf/2403.16591v3)

**Authors**: Xiaojin Zhang, Yulin Fei, Wei Chen

**Abstract**: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between LDP and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. The relationship between LDP and Maximum Bayesian Privacy (MBP) is first revealed, demonstrating that under uniform prior distribution, a mechanism satisfying $\xi$-LDP will satisfy $\xi$-MBP and conversely $\xi$-MBP also confers 2$\xi$-LDP. Our next theoretical contribution are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Maximum Bayesian Privacy (MBP), encapsulated by equations $\epsilon_{p,a} \leq \frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p,m} + \epsilon)\cdot(e^{\epsilon_{p,m} + \epsilon} - 1)}$. These relationships fortify our understanding of the privacy guarantees provided by various mechanisms. Our work not only lays the groundwork for future empirical exploration but also promises to facilitate the design of privacy-preserving algorithms, thereby fostering the development of trustworthy machine learning solutions.



## **20. PatchCURE: Improving Certifiable Robustness, Model Utility, and Computation Efficiency of Adversarial Patch Defenses**

cs.CV

USENIX Security 2024. (extended) technical report

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2310.13076v2) [paper-pdf](http://arxiv.org/pdf/2310.13076v2)

**Authors**: Chong Xiang, Tong Wu, Sihui Dai, Jonathan Petit, Suman Jana, Prateek Mittal

**Abstract**: State-of-the-art defenses against adversarial patch attacks can now achieve strong certifiable robustness with a marginal drop in model utility. However, this impressive performance typically comes at the cost of 10-100x more inference-time computation compared to undefended models -- the research community has witnessed an intense three-way trade-off between certifiable robustness, model utility, and computation efficiency. In this paper, we propose a defense framework named PatchCURE to approach this trade-off problem. PatchCURE provides sufficient "knobs" for tuning defense performance and allows us to build a family of defenses: the most robust PatchCURE instance can match the performance of any existing state-of-the-art defense (without efficiency considerations); the most efficient PatchCURE instance has similar inference efficiency as undefended models. Notably, PatchCURE achieves state-of-the-art robustness and utility performance across all different efficiency levels, e.g., 16-23% absolute clean accuracy and certified robust accuracy advantages over prior defenses when requiring computation efficiency to be close to undefended models. The family of PatchCURE defenses enables us to flexibly choose appropriate defenses to satisfy given computation and/or utility constraints in practice.



## **21. Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack**

cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01907v1) [paper-pdf](http://arxiv.org/pdf/2404.01907v1)

**Authors**: Ying Zhou, Ben He, Le Sun

**Abstract**: With the development of large language models (LLMs), detecting whether text is generated by a machine becomes increasingly challenging in the face of malicious use cases like the spread of false information, protection of intellectual property, and prevention of academic plagiarism. While well-trained text detectors have demonstrated promising performance on unseen test data, recent research suggests that these detectors have vulnerabilities when dealing with adversarial attacks such as paraphrasing. In this paper, we propose a framework for a broader class of adversarial attacks, designed to perform minor perturbations in machine-generated content to evade detection. We consider two attack settings: white-box and black-box, and employ adversarial learning in dynamic scenarios to assess the potential enhancement of the current detection model's robustness against such attacks. The empirical results reveal that the current detection models can be compromised in as little as 10 seconds, leading to the misclassification of machine-generated text as human-written content. Furthermore, we explore the prospect of improving the model's robustness over iterative adversarial learning. Although some improvements in model robustness are observed, practical applications still face significant challenges. These findings shed light on the future development of AI-text detectors, emphasizing the need for more accurate and robust detection methods.



## **22. Defense without Forgetting: Continual Adversarial Defense with Anisotropic & Isotropic Pseudo Replay**

cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01828v1) [paper-pdf](http://arxiv.org/pdf/2404.01828v1)

**Authors**: Yuhang Zhou, Zhongyun Hua

**Abstract**: Deep neural networks have demonstrated susceptibility to adversarial attacks. Adversarial defense techniques often focus on one-shot setting to maintain robustness against attack. However, new attacks can emerge in sequences in real-world deployment scenarios. As a result, it is crucial for a defense model to constantly adapt to new attacks, but the adaptation process can lead to catastrophic forgetting of previously defended against attacks. In this paper, we discuss for the first time the concept of continual adversarial defense under a sequence of attacks, and propose a lifelong defense baseline called Anisotropic \& Isotropic Replay (AIR), which offers three advantages: (1) Isotropic replay ensures model consistency in the neighborhood distribution of new data, indirectly aligning the output preference between old and new tasks. (2) Anisotropic replay enables the model to learn a compromise data manifold with fresh mixed semantics for further replay constraints and potential future attacks. (3) A straightforward regularizer mitigates the 'plasticity-stability' trade-off by aligning model output between new and old tasks. Experiment results demonstrate that AIR can approximate or even exceed the empirical performance upper bounds achieved by Joint Training.



## **23. Jailbreaking Prompt Attack: A Controllable Adversarial Attack against Diffusion Models**

cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02928v1) [paper-pdf](http://arxiv.org/pdf/2404.02928v1)

**Authors**: Jiachen Ma, Anda Cao, Zhiqing Xiao, Jie Zhang, Chao Ye, Junbo Zhao

**Abstract**: The fast advance of the image generation community has attracted attention worldwide. The safety issue needs to be further scrutinized and studied. There have been a few works around this area mostly achieving a post-processing design, model-specific, or yielding suboptimal image quality generation. Despite that, in this article, we discover a black-box attack method that enjoys three merits. It enables (i)-attacks both directed and semantic-driven that theoretically and practically pose a hazard to this vast user community, (ii)-surprisingly surpasses the white-box attack in a black-box manner and (iii)-without requiring any post-processing effort. Core to our approach is inspired by the concept guidance intriguing property of Classifier-Free guidance (CFG) in T2I models, and we discover that conducting frustratingly simple guidance in the CLIP embedding space, coupled with the semantic loss and an additionally sensitive word list works very well. Moreover, our results expose and highlight the vulnerabilities in existing defense mechanisms.



## **24. Security Allocation in Networked Control Systems under Stealthy Attacks**

eess.SY

12 pages, 3 figures, and 1 table, journal submission

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2308.16639v2) [paper-pdf](http://arxiv.org/pdf/2308.16639v2)

**Authors**: Anh Tung Nguyen, André M. H. Teixeira, Alexander Medvedev

**Abstract**: This paper considers the problem of security allocation in a networked control system under stealthy attacks. The system is comprised of interconnected subsystems represented by vertices. A malicious adversary selects a single vertex on which to conduct a stealthy data injection attack with the purpose of maximally disrupting a distant target vertex while remaining undetected. Defense resources against the adversary are allocated by a defender on several selected vertices. First, the objectives of the adversary and the defender with uncertain targets are formulated in a probabilistic manner, resulting in an expected worst-case impact of stealthy attacks. Next, we provide a graph-theoretic necessary and sufficient condition under which the cost for the defender and the expected worst-case impact of stealthy attacks are bounded. This condition enables the defender to restrict the admissible actions to dominating sets of the graph representing the network. Then, the security allocation problem is solved through a Stackelberg game-theoretic framework. Finally, the obtained results are validated through a numerical example of a 50-vertex networked control system.



## **25. ADVREPAIR:Provable Repair of Adversarial Attack**

cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01642v1) [paper-pdf](http://arxiv.org/pdf/2404.01642v1)

**Authors**: Zhiming Chi, Jianan Ma, Pengfei Yang, Cheng-Chao Huang, Renjue Li, Xiaowei Huang, Lijun Zhang

**Abstract**: Deep neural networks (DNNs) are increasingly deployed in safety-critical domains, but their vulnerability to adversarial attacks poses serious safety risks. Existing neuron-level methods using limited data lack efficacy in fixing adversaries due to the inherent complexity of adversarial attack mechanisms, while adversarial training, leveraging a large number of adversarial samples to enhance robustness, lacks provability. In this paper, we propose ADVREPAIR, a novel approach for provable repair of adversarial attacks using limited data. By utilizing formal verification, ADVREPAIR constructs patch modules that, when integrated with the original network, deliver provable and specialized repairs within the robustness neighborhood. Additionally, our approach incorporates a heuristic mechanism for assigning patch modules, allowing this defense against adversarial attacks to generalize to other inputs. ADVREPAIR demonstrates superior efficiency, scalability and repair success rate. Different from existing DNN repair methods, our repair can generalize to general inputs, thereby improving the robustness of the neural network globally, which indicates a significant breakthrough in the generalization capability of ADVREPAIR.



## **26. Multi-granular Adversarial Attacks against Black-box Neural Ranking Models**

cs.IR

Accepted by SIGIR 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01574v1) [paper-pdf](http://arxiv.org/pdf/2404.01574v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Adversarial ranking attacks have gained increasing attention due to their success in probing vulnerabilities, and, hence, enhancing the robustness, of neural ranking models. Conventional attack methods employ perturbations at a single granularity, e.g., word-level or sentence-level, to a target document. However, limiting perturbations to a single level of granularity may reduce the flexibility of creating adversarial examples, thereby diminishing the potential threat of the attack. Therefore, we focus on generating high-quality adversarial examples by incorporating multi-granular perturbations. Achieving this objective involves tackling a combinatorial explosion problem, which requires identifying an optimal combination of perturbations across all possible levels of granularity, positions, and textual pieces. To address this challenge, we transform the multi-granular adversarial attack into a sequential decision-making process, where perturbations in the next attack step are influenced by the perturbed document in the current attack step. Since the attack process can only access the final state without direct intermediate signals, we use reinforcement learning to perform multi-granular attacks. During the reinforcement learning process, two agents work cooperatively to identify multi-granular vulnerabilities as attack targets and organize perturbation candidates into a final perturbation sequence. Experimental results show that our attack method surpasses prevailing baselines in both attack effectiveness and imperceptibility.



## **27. MMCert: Provable Defense against Adversarial Attacks to Multi-modal Models**

cs.CV

To appear in CVPR'24

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.19080v3) [paper-pdf](http://arxiv.org/pdf/2403.19080v3)

**Authors**: Yanting Wang, Hongye Fu, Wei Zou, Jinyuan Jia

**Abstract**: Different from a unimodal model whose input is from a single modality, the input (called multi-modal input) of a multi-modal model is from multiple modalities such as image, 3D points, audio, text, etc. Similar to unimodal models, many existing studies show that a multi-modal model is also vulnerable to adversarial perturbation, where an attacker could add small perturbation to all modalities of a multi-modal input such that the multi-modal model makes incorrect predictions for it. Existing certified defenses are mostly designed for unimodal models, which achieve sub-optimal certified robustness guarantees when extended to multi-modal models as shown in our experimental results. In our work, we propose MMCert, the first certified defense against adversarial attacks to a multi-modal model. We derive a lower bound on the performance of our MMCert under arbitrary adversarial attacks with bounded perturbations to both modalities (e.g., in the context of auto-driving, we bound the number of changed pixels in both RGB image and depth image). We evaluate our MMCert using two benchmark datasets: one for the multi-modal road segmentation task and the other for the multi-modal emotion recognition task. Moreover, we compare our MMCert with a state-of-the-art certified defense extended from unimodal models. Our experimental results show that our MMCert outperforms the baseline.



## **28. Rumor Detection with a novel graph neural network approach**

cs.AI

10 pages, 5 figures

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.16206v3) [paper-pdf](http://arxiv.org/pdf/2403.16206v3)

**Authors**: Tianrui Liu, Qi Cai, Changxin Xu, Bo Hong, Fanghao Ni, Yuxin Qiao, Tsungwei Yang

**Abstract**: The wide spread of rumors on social media has caused a negative impact on people's daily life, leading to potential panic, fear, and mental health problems for the public. How to debunk rumors as early as possible remains a challenging problem. Existing studies mainly leverage information propagation structure to detect rumors, while very few works focus on correlation among users that they may coordinate to spread rumors in order to gain large popularity. In this paper, we propose a new detection model, that jointly learns both the representations of user correlation and information propagation to detect rumors on social media. Specifically, we leverage graph neural networks to learn the representations of user correlation from a bipartite graph that describes the correlations between users and source tweets, and the representations of information propagation with a tree structure. Then we combine the learned representations from these two modules to classify the rumors. Since malicious users intend to subvert our model after deployment, we further develop a greedy attack scheme to analyze the cost of three adversarial attacks: graph attack, comment attack, and joint attack. Evaluation results on two public datasets illustrate that the proposed MODEL outperforms the state-of-the-art rumor detection models. We also demonstrate our method performs well for early rumor detection. Moreover, the proposed detection method is more robust to adversarial attacks compared to the best existing method. Importantly, we show that it requires a high cost for attackers to subvert user correlation pattern, demonstrating the importance of considering user correlation for rumor detection.



## **29. Vulnerabilities of Foundation Model Integrated Federated Learning Under Adversarial Threats**

cs.CR

Chen Wu and Xi Li are equal contribution. The corresponding author is  Jiaqi Wang

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2401.10375v2) [paper-pdf](http://arxiv.org/pdf/2401.10375v2)

**Authors**: Chen Wu, Xi Li, Jiaqi Wang

**Abstract**: Federated Learning (FL) addresses critical issues in machine learning related to data privacy and security, yet suffering from data insufficiency and imbalance under certain circumstances. The emergence of foundation models (FMs) offers potential solutions to the limitations of existing FL frameworks, e.g., by generating synthetic data for model initialization. However, due to the inherent safety concerns of FMs, integrating FMs into FL could introduce new risks, which remains largely unexplored. To address this gap, we conduct the first investigation on the vulnerability of FM integrated FL (FM-FL) under adversarial threats. Based on a unified framework of FM-FL, we introduce a novel attack strategy that exploits safety issues of FM to compromise FL client models. Through extensive experiments with well-known models and benchmark datasets in both image and text domains, we reveal the high susceptibility of the FM-FL to this new threat under various FL configurations. Furthermore, we find that existing FL defense strategies offer limited protection against this novel attack approach. This research highlights the critical need for enhanced security measures in FL in the era of FMs.



## **30. Can Biases in ImageNet Models Explain Generalization?**

cs.CV

Accepted at CVPR2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01509v1) [paper-pdf](http://arxiv.org/pdf/2404.01509v1)

**Authors**: Paul Gavrikov, Janis Keuper

**Abstract**: The robust generalization of models to rare, in-distribution (ID) samples drawn from the long tail of the training distribution and to out-of-training-distribution (OOD) samples is one of the major challenges of current deep learning methods. For image classification, this manifests in the existence of adversarial attacks, the performance drops on distorted images, and a lack of generalization to concepts such as sketches. The current understanding of generalization in neural networks is very limited, but some biases that differentiate models from human vision have been identified and might be causing these limitations. Consequently, several attempts with varying success have been made to reduce these biases during training to improve generalization. We take a step back and sanity-check these attempts. Fixing the architecture to the well-established ResNet-50, we perform a large-scale study on 48 ImageNet models obtained via different training methods to understand how and if these biases - including shape bias, spectral biases, and critical bands - interact with generalization. Our extensive study results reveal that contrary to previous findings, these biases are insufficient to accurately predict the generalization of a model holistically. We provide access to all checkpoints and evaluation code at https://github.com/paulgavrikov/biases_vs_generalization



## **31. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

cs.CR

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2403.05030v3) [paper-pdf](http://arxiv.org/pdf/2403.05030v3)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use it to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.



## **32. Robust One-Class Classification with Signed Distance Function using 1-Lipschitz Neural Networks**

cs.LG

27 pages, 11 figures, International Conference on Machine Learning  2023, (ICML 2023)

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2303.01978v2) [paper-pdf](http://arxiv.org/pdf/2303.01978v2)

**Authors**: Louis Bethune, Paul Novello, Thibaut Boissin, Guillaume Coiffier, Mathieu Serrurier, Quentin Vincenot, Andres Troya-Galvis

**Abstract**: We propose a new method, dubbed One Class Signed Distance Function (OCSDF), to perform One Class Classification (OCC) by provably learning the Signed Distance Function (SDF) to the boundary of the support of any distribution. The distance to the support can be interpreted as a normality score, and its approximation using 1-Lipschitz neural networks provides robustness bounds against $l2$ adversarial attacks, an under-explored weakness of deep learning-based OCC algorithms. As a result, OCSDF comes with a new metric, certified AUROC, that can be computed at the same cost as any classical AUROC. We show that OCSDF is competitive against concurrent methods on tabular and image data while being way more robust to adversarial attacks, illustrating its theoretical properties. Finally, as exploratory research perspectives, we theoretically and empirically show how OCSDF connects OCC with image generation and implicit neural surface parametrization. Our code is available at https://github.com/Algue-Rythme/OneClassMetricLearning



## **33. The twin peaks of learning neural networks**

cs.LG

37 pages, 31 figures

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2401.12610v2) [paper-pdf](http://arxiv.org/pdf/2401.12610v2)

**Authors**: Elizaveta Demyanenko, Christoph Feinauer, Enrico M. Malatesta, Luca Saglietti

**Abstract**: Recent works demonstrated the existence of a double-descent phenomenon for the generalization error of neural networks, where highly overparameterized models escape overfitting and achieve good test performance, at odds with the standard bias-variance trade-off described by statistical learning theory. In the present work, we explore a link between this phenomenon and the increase of complexity and sensitivity of the function represented by neural networks. In particular, we study the Boolean mean dimension (BMD), a metric developed in the context of Boolean function analysis. Focusing on a simple teacher-student setting for the random feature model, we derive a theoretical analysis based on the replica method that yields an interpretable expression for the BMD, in the high dimensional regime where the number of data points, the number of features, and the input size grow to infinity. We find that, as the degree of overparameterization of the network is increased, the BMD reaches an evident peak at the interpolation threshold, in correspondence with the generalization error peak, and then slowly approaches a low asymptotic value. The same phenomenology is then traced in numerical experiments with different model classes and training setups. Moreover, we find empirically that adversarially initialized models tend to show higher BMD values, and that models that are more robust to adversarial attacks exhibit a lower BMD.



## **34. Foundations of Cyber Resilience: The Confluence of Game, Control, and Learning Theories**

eess.SY

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01205v1) [paper-pdf](http://arxiv.org/pdf/2404.01205v1)

**Authors**: Quanyan Zhu

**Abstract**: Cyber resilience is a complementary concept to cybersecurity, focusing on the preparation, response, and recovery from cyber threats that are challenging to prevent. Organizations increasingly face such threats in an evolving cyber threat landscape. Understanding and establishing foundations for cyber resilience provide a quantitative and systematic approach to cyber risk assessment, mitigation policy evaluation, and risk-informed defense design. A systems-scientific view toward cyber risks provides holistic and system-level solutions. This chapter starts with a systemic view toward cyber risks and presents the confluence of game theory, control theory, and learning theories, which are three major pillars for the design of cyber resilience mechanisms to counteract increasingly sophisticated and evolving threats in our networks and organizations. Game and control theoretic methods provide a set of modeling frameworks to capture the strategic and dynamic interactions between defenders and attackers. Control and learning frameworks together provide a feedback-driven mechanism that enables autonomous and adaptive responses to threats. Game and learning frameworks offer a data-driven approach to proactively reason about adversarial behaviors and resilient strategies. The confluence of the three lays the theoretical foundations for the analysis and design of cyber resilience. This chapter presents various theoretical paradigms, including dynamic asymmetric games, moving horizon control, conjectural learning, and meta-learning, as recent advances at the intersection. This chapter concludes with future directions and discussions of the role of neurosymbolic learning and the synergy between foundation models and game models in cyber resilience.



## **35. The Best Defense is Attack: Repairing Semantics in Textual Adversarial Examples**

cs.CL

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2305.04067v2) [paper-pdf](http://arxiv.org/pdf/2305.04067v2)

**Authors**: Heng Yang, Ke Li

**Abstract**: Recent studies have revealed the vulnerability of pre-trained language models to adversarial attacks. Existing adversarial defense techniques attempt to reconstruct adversarial examples within feature or text spaces. However, these methods struggle to effectively repair the semantics in adversarial examples, resulting in unsatisfactory performance and limiting their practical utility. To repair the semantics in adversarial examples, we introduce a novel approach named Reactive Perturbation Defocusing (Rapid). Rapid employs an adversarial detector to identify fake labels of adversarial examples and leverage adversarial attackers to repair the semantics in adversarial examples. Our extensive experimental results conducted on four public datasets, convincingly demonstrate the effectiveness of Rapid in various adversarial attack scenarios. To address the problem of defense performance validation in previous works, we provide a demonstration of adversarial detection and repair based on our work, which can be easily evaluated at https://tinyurl.com/22ercuf8.



## **36. Poisoning Decentralized Collaborative Recommender System and Its Countermeasures**

cs.CR

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01177v1) [paper-pdf](http://arxiv.org/pdf/2404.01177v1)

**Authors**: Ruiqi Zheng, Liang Qu, Tong Chen, Kai Zheng, Yuhui Shi, Hongzhi Yin

**Abstract**: To make room for privacy and efficiency, the deployment of many recommender systems is experiencing a shift from central servers to personal devices, where the federated recommender systems (FedRecs) and decentralized collaborative recommender systems (DecRecs) are arguably the two most representative paradigms. While both leverage knowledge (e.g., gradients) sharing to facilitate learning local models, FedRecs rely on a central server to coordinate the optimization process, yet in DecRecs, the knowledge sharing directly happens between clients. Knowledge sharing also opens a backdoor for model poisoning attacks, where adversaries disguise themselves as benign clients and disseminate polluted knowledge to achieve malicious goals like promoting an item's exposure rate. Although research on such poisoning attacks provides valuable insights into finding security loopholes and corresponding countermeasures, existing attacks mostly focus on FedRecs, and are either inapplicable or ineffective for DecRecs. Compared with FedRecs where the tampered information can be universally distributed to all clients once uploaded to the cloud, each adversary in DecRecs can only communicate with neighbor clients of a small size, confining its impact to a limited range. To fill the gap, we present a novel attack method named Poisoning with Adaptive Malicious Neighbors (PAMN). With item promotion in top-K recommendation as the attack objective, PAMN effectively boosts target items' ranks with several adversaries that emulate benign clients and transfers adaptively crafted gradients conditioned on each adversary's neighbors. Moreover, with the vulnerabilities of DecRecs uncovered, a dedicated defensive mechanism based on user-level gradient clipping with sparsified updating is proposed. Extensive experiments demonstrate the effectiveness of the poisoning attack and the robustness of our defensive mechanism.



## **37. The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness**

cs.LG

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01356v1) [paper-pdf](http://arxiv.org/pdf/2404.01356v1)

**Authors**: Xuran Li, Peng Wu, Yanting Chen, Xingjun Ma, Zhen Zhang, Kaixiang Dong

**Abstract**: Deep neural networks (DNNs) are known to be sensitive to adversarial input perturbations, leading to a reduction in either prediction accuracy or individual fairness. To jointly characterize the susceptibility of prediction accuracy and individual fairness to adversarial perturbations, we introduce a novel robustness definition termed robust accurate fairness. Informally, robust accurate fairness requires that predictions for an instance and its similar counterparts consistently align with the ground truth when subjected to input perturbations. We propose an adversarial attack approach dubbed RAFair to expose false or biased adversarial defects in DNN, which either deceive accuracy or compromise individual fairness. Then, we show that such adversarial instances can be effectively addressed by carefully designed benign perturbations, correcting their predictions to be accurate and fair. Our work explores the double-edged sword of input perturbations to robust accurate fairness in DNN and the potential of using benign perturbations to correct adversarial instances.



## **38. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

cs.CL

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2310.00322v3) [paper-pdf](http://arxiv.org/pdf/2310.00322v3)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.



## **39. LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer**

cs.CV

14 pages, 3 figures. Accepted to AAAI 2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2312.09935v2) [paper-pdf](http://arxiv.org/pdf/2312.09935v2)

**Authors**: Yuxin Cao, Ziyu Zhao, Xi Xiao, Derui Wang, Minhui Xue, Jin Lu

**Abstract**: Video recognition systems are vulnerable to adversarial examples. Recent studies show that style transfer-based and patch-based unrestricted perturbations can effectively improve attack efficiency. These attacks, however, face two main challenges: 1) Adding large stylized perturbations to all pixels reduces the naturalness of the video and such perturbations can be easily detected. 2) Patch-based video attacks are not extensible to targeted attacks due to the limited search space of reinforcement learning that has been widely used in video attacks recently. In this paper, we focus on the video black-box setting and propose a novel attack framework named LogoStyleFool by adding a stylized logo to the clean video. We separate the attack into three stages: style reference selection, reinforcement-learning-based logo style transfer, and perturbation optimization. We solve the first challenge by scaling down the perturbation range to a regional logo, while the second challenge is addressed by complementing an optimization stage after reinforcement learning. Experimental results substantiate the overall superiority of LogoStyleFool over three state-of-the-art patch-based attacks in terms of attack performance and semantic preservation. Meanwhile, LogoStyleFool still maintains its performance against two existing patch-based defense methods. We believe that our research is beneficial in increasing the attention of the security community to such subregional style transfer attacks.



## **40. StyleFool: Fooling Video Classification Systems via Style Transfer**

cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2203.16000v4) [paper-pdf](http://arxiv.org/pdf/2203.16000v4)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.



## **41. BadPart: Unified Black-box Adversarial Patch Attacks against Pixel-wise Regression Tasks**

cs.CV

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.00924v1) [paper-pdf](http://arxiv.org/pdf/2404.00924v1)

**Authors**: Zhiyuan Cheng, Zhaoyi Liu, Tengda Guo, Shiwei Feng, Dongfang Liu, Mingjie Tang, Xiangyu Zhang

**Abstract**: Pixel-wise regression tasks (e.g., monocular depth estimation (MDE) and optical flow estimation (OFE)) have been widely involved in our daily life in applications like autonomous driving, augmented reality and video composition. Although certain applications are security-critical or bear societal significance, the adversarial robustness of such models are not sufficiently studied, especially in the black-box scenario. In this work, we introduce the first unified black-box adversarial patch attack framework against pixel-wise regression tasks, aiming to identify the vulnerabilities of these models under query-based black-box attacks. We propose a novel square-based adversarial patch optimization framework and employ probabilistic square sampling and score-based gradient estimation techniques to generate the patch effectively and efficiently, overcoming the scalability problem of previous black-box patch attacks. Our attack prototype, named BadPart, is evaluated on both MDE and OFE tasks, utilizing a total of 7 models. BadPart surpasses 3 baseline methods in terms of both attack performance and efficiency. We also apply BadPart on the Google online service for portrait depth estimation, causing 43.5% relative distance error with 50K queries. State-of-the-art (SOTA) countermeasures cannot defend our attack effectively.



## **42. An Embarrassingly Simple Defense Against Backdoor Attacks On SSL**

cs.CV

10 pages, 5 figures

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2403.15918v2) [paper-pdf](http://arxiv.org/pdf/2403.15918v2)

**Authors**: Aryan Satpathy, Nilaksh Nilaksh, Dhruva Rajwade

**Abstract**: Self Supervised Learning (SSL) has emerged as a powerful paradigm to tackle data landscapes with absence of human supervision. The ability to learn meaningful tasks without the use of labeled data makes SSL a popular method to manage large chunks of data in the absence of labels. However, recent work indicates SSL to be vulnerable to backdoor attacks, wherein models can be controlled, possibly maliciously, to suit an adversary's motives. Li et. al (2022) introduce a novel frequency-based backdoor attack: CTRL. They show that CTRL can be used to efficiently and stealthily gain control over a victim's model trained using SSL. In this work, we devise two defense strategies against frequency-based attacks in SSL: One applicable before model training and the second to be applied during model inference. Our first contribution utilizes the invariance property of the downstream task to defend against backdoor attacks in a generalizable fashion. We observe the ASR (Attack Success Rate) to reduce by over 60% across experiments. Our Inference-time defense relies on evasiveness of the attack and uses the luminance channel to defend against attacks. Using object classification as the downstream task for SSL, we demonstrate successful defense strategies that do not require re-training of the model. Code is available at https://github.com/Aryan-Satpathy/Backdoor.



## **43. Machine Learning Robustness: A Primer**

cs.LG

arXiv admin note: text overlap with arXiv:2305.10862 by other authors

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.00897v1) [paper-pdf](http://arxiv.org/pdf/2404.00897v1)

**Authors**: Houssem Ben Braiek, Foutse Khomh

**Abstract**: This chapter explores the foundational concept of robustness in Machine Learning (ML) and its integral role in establishing trustworthiness in Artificial Intelligence (AI) systems. The discussion begins with a detailed definition of robustness, portraying it as the ability of ML models to maintain stable performance across varied and unexpected environmental conditions. ML robustness is dissected through several lenses: its complementarity with generalizability; its status as a requirement for trustworthy AI; its adversarial vs non-adversarial aspects; its quantitative metrics; and its indicators such as reproducibility and explainability. The chapter delves into the factors that impede robustness, such as data bias, model complexity, and the pitfalls of underspecified ML pipelines. It surveys key techniques for robustness assessment from a broad perspective, including adversarial attacks, encompassing both digital and physical realms. It covers non-adversarial data shifts and nuances of Deep Learning (DL) software testing methodologies. The discussion progresses to explore amelioration strategies for bolstering robustness, starting with data-centric approaches like debiasing and augmentation. Further examination includes a variety of model-centric methods such as transfer learning, adversarial training, and randomized smoothing. Lastly, post-training methods are discussed, including ensemble techniques, pruning, and model repairs, emerging as cost-effective strategies to make models more resilient against the unpredictable. This chapter underscores the ongoing challenges and limitations in estimating and achieving ML robustness by existing approaches. It offers insights and directions for future research on this crucial concept, as a prerequisite for trustworthy AI systems.



## **44. Privacy Re-identification Attacks on Tabular GANs**

cs.CR

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2404.00696v1) [paper-pdf](http://arxiv.org/pdf/2404.00696v1)

**Authors**: Abdallah Alshantti, Adil Rasheed, Frank Westad

**Abstract**: Generative models are subject to overfitting and thus may potentially leak sensitive information from the training data. In this work. we investigate the privacy risks that can potentially arise from the use of generative adversarial networks (GANs) for creating tabular synthetic datasets. For the purpose, we analyse the effects of re-identification attacks on synthetic data, i.e., attacks which aim at selecting samples that are predicted to correspond to memorised training samples based on their proximity to the nearest synthetic records. We thus consider multiple settings where different attackers might have different access levels or knowledge of the generative model and predictive, and assess which information is potentially most useful for launching more successful re-identification attacks. In doing so we also consider the situation for which re-identification attacks are formulated as reconstruction attacks, i.e., the situation where an attacker uses evolutionary multi-objective optimisation for perturbing synthetic samples closer to the training space. The results indicate that attackers can indeed pose major privacy risks by selecting synthetic samples that are likely representative of memorised training samples. In addition, we notice that privacy threats considerably increase when the attacker either has knowledge or has black-box access to the generative models. We also find that reconstruction attacks through multi-objective optimisation even increase the risk of identifying confidential samples.



## **45. Model-less Is the Best Model: Generating Pure Code Implementations to Replace On-Device DL Models**

cs.SE

Accepted by the ACM SIGSOFT International Symposium on Software  Testing and Analysis (ISSTA2024)

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2403.16479v2) [paper-pdf](http://arxiv.org/pdf/2403.16479v2)

**Authors**: Mingyi Zhou, Xiang Gao, Pei Liu, John Grundy, Chunyang Chen, Xiao Chen, Li Li

**Abstract**: Recent studies show that deployed deep learning (DL) models such as those of Tensor Flow Lite (TFLite) can be easily extracted from real-world applications and devices by attackers to generate many kinds of attacks like adversarial attacks. Although securing deployed on-device DL models has gained increasing attention, no existing methods can fully prevent the aforementioned threats. Traditional software protection techniques have been widely explored, if on-device models can be implemented using pure code, such as C++, it will open the possibility of reusing existing software protection techniques. However, due to the complexity of DL models, there is no automatic method that can translate the DL models to pure code. To fill this gap, we propose a novel method, CustomDLCoder, to automatically extract the on-device model information and synthesize a customized executable program for a wide range of DL models. CustomDLCoder first parses the DL model, extracts its backend computing units, configures the computing units to a graph, and then generates customized code to implement and deploy the ML solution without explicit model representation. The synthesized program hides model information for DL deployment environments since it does not need to retain explicit model representation, preventing many attacks on the DL model. In addition, it improves ML performance because the customized code removes model parsing and preprocessing steps and only retains the data computing process. Our experimental results show that CustomDLCoder improves model security by disabling on-device model sniffing. Compared with the original on-device platform (i.e., TFLite), our method can accelerate model inference by 21.8% and 24.3% on x86-64 and ARM64 platforms, respectively. Most importantly, it can significantly reduce memory consumption by 68.8% and 36.0% on x86-64 and ARM64 platforms, respectively.



## **46. Embodied Active Defense: Leveraging Recurrent Feedback to Counter Adversarial Patches**

cs.CV

27pages

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2404.00540v1) [paper-pdf](http://arxiv.org/pdf/2404.00540v1)

**Authors**: Lingxuan Wu, Xiao Yang, Yinpeng Dong, Liuwei Xie, Hang Su, Jun Zhu

**Abstract**: The vulnerability of deep neural networks to adversarial patches has motivated numerous defense strategies for boosting model robustness. However, the prevailing defenses depend on single observation or pre-established adversary information to counter adversarial patches, often failing to be confronted with unseen or adaptive adversarial attacks and easily exhibiting unsatisfying performance in dynamic 3D environments. Inspired by active human perception and recurrent feedback mechanisms, we develop Embodied Active Defense (EAD), a proactive defensive strategy that actively contextualizes environmental information to address misaligned adversarial patches in 3D real-world settings. To achieve this, EAD develops two central recurrent sub-modules, i.e., a perception module and a policy module, to implement two critical functions of active vision. These models recurrently process a series of beliefs and observations, facilitating progressive refinement of their comprehension of the target object and enabling the development of strategic actions to counter adversarial patches in 3D environments. To optimize learning efficiency, we incorporate a differentiable approximation of environmental dynamics and deploy patches that are agnostic to the adversary strategies. Extensive experiments demonstrate that EAD substantially enhances robustness against a variety of patches within just a few steps through its action policy in safety-critical tasks (e.g., face recognition and object detection), without compromising standard accuracy. Furthermore, due to the attack-agnostic characteristic, EAD facilitates excellent generalization to unseen attacks, diminishing the averaged attack success rate by 95 percent across a range of unseen adversarial attacks.



## **47. An Unsupervised Adversarial Autoencoder for Cyber Attack Detection in Power Distribution Grids**

cs.CR

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2404.02923v1) [paper-pdf](http://arxiv.org/pdf/2404.02923v1)

**Authors**: Mehdi Jabbari Zideh, Mohammad Reza Khalghani, Sarika Khushalani Solanki

**Abstract**: Detection of cyber attacks in smart power distribution grids with unbalanced configurations poses challenges due to the inherent nonlinear nature of these uncertain and stochastic systems. It originates from the intermittent characteristics of the distributed energy resources (DERs) generation and load variations. Moreover, the unknown behavior of cyber attacks, especially false data injection attacks (FDIAs) in the distribution grids with complex temporal correlations and the limited amount of labeled data increases the vulnerability of the grids and imposes a high risk in the secure and reliable operation of the grids. To address these challenges, this paper proposes an unsupervised adversarial autoencoder (AAE) model to detect FDIAs in unbalanced power distribution grids integrated with DERs, i.e., PV systems and wind generation. The proposed method utilizes long short-term memory (LSTM) in the structure of the autoencoder to capture the temporal dependencies in the time-series measurements and leverages the power of generative adversarial networks (GANs) for better reconstruction of the input data. The advantage of the proposed data-driven model is that it can detect anomalous points for the system operation without reliance on abstract models or mathematical representations. To evaluate the efficacy of the approach, it is tested on IEEE 13-bus and 123-bus systems with historical meteorological data (wind speed, ambient temperature, and solar irradiance) as well as historical real-world load data under three types of data falsification functions. The comparison of the detection results of the proposed model with other unsupervised learning methods verifies its superior performance in detecting cyber attacks in unbalanced power distribution grids.



## **48. AttackNet: Enhancing Biometric Security via Tailored Convolutional Neural Network Architectures for Liveness Detection**

cs.CV

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2402.03769v2) [paper-pdf](http://arxiv.org/pdf/2402.03769v2)

**Authors**: Oleksandr Kuznetsov, Dmytro Zakharov, Emanuele Frontoni, Andrea Maranesi

**Abstract**: Biometric security is the cornerstone of modern identity verification and authentication systems, where the integrity and reliability of biometric samples is of paramount importance. This paper introduces AttackNet, a bespoke Convolutional Neural Network architecture, meticulously designed to combat spoofing threats in biometric systems. Rooted in deep learning methodologies, this model offers a layered defense mechanism, seamlessly transitioning from low-level feature extraction to high-level pattern discernment. Three distinctive architectural phases form the crux of the model, each underpinned by judiciously chosen activation functions, normalization techniques, and dropout layers to ensure robustness and resilience against adversarial attacks. Benchmarking our model across diverse datasets affirms its prowess, showcasing superior performance metrics in comparison to contemporary models. Furthermore, a detailed comparative analysis accentuates the model's efficacy, drawing parallels with prevailing state-of-the-art methodologies. Through iterative refinement and an informed architectural strategy, AttackNet underscores the potential of deep learning in safeguarding the future of biometric security.



## **49. Bidirectional Consistency Models**

cs.LG

40 pages, 25 figures

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2403.18035v2) [paper-pdf](http://arxiv.org/pdf/2403.18035v2)

**Authors**: Liangchen Li, Jiajun He

**Abstract**: Diffusion models (DMs) are capable of generating remarkably high-quality samples by iteratively denoising a random vector, a process that corresponds to moving along the probability flow ordinary differential equation (PF ODE). Interestingly, DMs can also invert an input image to noise by moving backward along the PF ODE, a key operation for downstream tasks such as interpolation and image editing. However, the iterative nature of this process restricts its speed, hindering its broader application. Recently, Consistency Models (CMs) have emerged to address this challenge by approximating the integral of the PF ODE, largely reducing the number of iterations. Yet, the absence of an explicit ODE solver complicates the inversion process. To resolve this, we introduce the Bidirectional Consistency Model (BCM), which learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation and inversion tasks within one framework. Notably, our proposed method enables one-step generation and inversion while also allowing the use of additional steps to enhance generation quality or reduce reconstruction error. Furthermore, by leveraging our model's bidirectional consistency, we introduce a sampling strategy that can enhance FID while preserving the generated image content. We further showcase our model's capabilities in several downstream tasks, such as interpolation and inpainting, and present demonstrations of potential applications, including blind restoration of compressed images and defending black-box adversarial attacks.



## **50. STBA: Towards Evaluating the Robustness of DNNs for Query-Limited Black-box Scenario**

cs.CV

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2404.00362v1) [paper-pdf](http://arxiv.org/pdf/2404.00362v1)

**Authors**: Renyang Liu, Kwok-Yan Lam, Wei Zhou, Sixing Wu, Jun Zhao, Dongting Hu, Mingming Gong

**Abstract**: Many attack techniques have been proposed to explore the vulnerability of DNNs and further help to improve their robustness. Despite the significant progress made recently, existing black-box attack methods still suffer from unsatisfactory performance due to the vast number of queries needed to optimize desired perturbations. Besides, the other critical challenge is that adversarial examples built in a noise-adding manner are abnormal and struggle to successfully attack robust models, whose robustness is enhanced by adversarial training against small perturbations. There is no doubt that these two issues mentioned above will significantly increase the risk of exposure and result in a failure to dig deeply into the vulnerability of DNNs. Hence, it is necessary to evaluate DNNs' fragility sufficiently under query-limited settings in a non-additional way. In this paper, we propose the Spatial Transform Black-box Attack (STBA), a novel framework to craft formidable adversarial examples in the query-limited scenario. Specifically, STBA introduces a flow field to the high-frequency part of clean images to generate adversarial examples and adopts the following two processes to enhance their naturalness and significantly improve the query efficiency: a) we apply an estimated flow field to the high-frequency part of clean images to generate adversarial examples instead of introducing external noise to the benign image, and b) we leverage an efficient gradient estimation method based on a batch of samples to optimize such an ideal flow field under query-limited settings. Compared to existing score-based black-box baselines, extensive experiments indicated that STBA could effectively improve the imperceptibility of the adversarial examples and remarkably boost the attack success rate under query-limited settings.



