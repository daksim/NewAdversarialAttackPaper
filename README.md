# Latest Adversarial Attack Papers
**update at 2024-07-31 16:20:58**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20836v1) [paper-pdf](http://arxiv.org/pdf/2407.20836v1)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of these AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. For the task of AIGI detection, we propose a new attack containing two main parts. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous models, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as frequency-based post-train Bayesian attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario.



## **2. Attacking Cooperative Multi-Agent Reinforcement Learning by Adversarial Minority Influence**

cs.LG

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2302.03322v3) [paper-pdf](http://arxiv.org/pdf/2302.03322v3)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Yuwei Zheng, Pu Feng, Xin Yu, Aishan Liu, Yaodong Yang, Bo An, Wenjun Wu, Xianglong Liu

**Abstract**: This study probes the vulnerabilities of cooperative multi-agent reinforcement learning (c-MARL) under adversarial attacks, a critical determinant of c-MARL's worst-case performance prior to real-world implementation. Current observation-based attacks, constrained by white-box assumptions, overlook c-MARL's complex multi-agent interactions and cooperative objectives, resulting in impractical and limited attack capabilities. To address these shortcomes, we propose Adversarial Minority Influence (AMI), a practical and strong for c-MARL. AMI is a practical black-box attack and can be launched without knowing victim parameters. AMI is also strong by considering the complex multi-agent interaction and the cooperative goal of agents, enabling a single adversarial agent to unilaterally misleads majority victims to form targeted worst-case cooperation. This mirrors minority influence phenomena in social psychology. To achieve maximum deviation in victim policies under complex agent-wise interactions, our unilateral attack aims to characterize and maximize the impact of the adversary on the victims. This is achieved by adapting a unilateral agent-wise relation metric derived from mutual information, thereby mitigating the adverse effects of victim influence on the adversary. To lead the victims into a jointly detrimental scenario, our targeted attack deceives victims into a long-term, cooperatively harmful situation by guiding each victim towards a specific target, determined through a trial-and-error process executed by a reinforcement learning agent. Through AMI, we achieve the first successful attack against real-world robot swarms and effectively fool agents in simulated environments into collectively worst-case scenarios, including Starcraft II and Multi-agent Mujoco. The source code and demonstrations can be found at: https://github.com/DIG-Beihang/AMI.



## **3. Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks**

cs.CV

Accepted to ECCV 2024, Project Page: https://PDCL-Attack.github.io

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20657v1) [paper-pdf](http://arxiv.org/pdf/2407.20657v1)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Recent vision-language foundation models, such as CLIP, have demonstrated superior capabilities in learning representations that can be transferable across diverse range of downstream tasks and domains. With the emergence of such powerful models, it has become crucial to effectively leverage their capabilities in tackling challenging vision tasks. On the other hand, only a few works have focused on devising adversarial examples that transfer well to both unknown domains and model architectures. In this paper, we propose a novel transfer attack method called PDCL-Attack, which leverages the CLIP model to enhance the transferability of adversarial perturbations generated by a generative model-based attack framework. Specifically, we formulate an effective prompt-driven feature guidance by harnessing the semantic representation power of text, particularly from the ground-truth class labels of input images. To the best of our knowledge, we are the first to introduce prompt learning to enhance the transferable generative attacks. Extensive experiments conducted across various cross-domain and cross-model settings empirically validate our approach, demonstrating its superiority over state-of-the-art methods.



## **4. FACL-Attack: Frequency-Aware Contrastive Learning for Transferable Adversarial Attacks**

cs.CV

Accepted to AAAI 2024, Project Page: https://FACL-Attack.github.io

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20653v1) [paper-pdf](http://arxiv.org/pdf/2407.20653v1)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Deep neural networks are known to be vulnerable to security risks due to the inherent transferable nature of adversarial examples. Despite the success of recent generative model-based attacks demonstrating strong transferability, it still remains a challenge to design an efficient attack strategy in a real-world strict black-box setting, where both the target domain and model architectures are unknown. In this paper, we seek to explore a feature contrastive approach in the frequency domain to generate adversarial examples that are robust in both cross-domain and cross-model settings. With that goal in mind, we propose two modules that are only employed during the training phase: a Frequency-Aware Domain Randomization (FADR) module to randomize domain-variant low- and high-range frequency components and a Frequency-Augmented Contrastive Learning (FACL) module to effectively separate domain-invariant mid-frequency features of clean and perturbed image. We demonstrate strong transferability of our generated adversarial perturbations through extensive cross-domain and cross-model experiments, while keeping the inference time complexity.



## **5. Robust Federated Learning for Wireless Networks: A Demonstration with Channel Estimation**

cs.LG

Submitted to IEEE GLOBECOM 2024

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2404.03088v2) [paper-pdf](http://arxiv.org/pdf/2404.03088v2)

**Authors**: Zexin Fang, Bin Han, Hans D. Schotten

**Abstract**: Federated learning (FL) offers a privacy-preserving collaborative approach for training models in wireless networks, with channel estimation emerging as a promising application. Despite extensive studies on FL-empowered channel estimation, the security concerns associated with FL require meticulous attention. In a scenario where small base stations (SBSs) serve as local models trained on cached data, and a macro base station (MBS) functions as the global model setting, an attacker can exploit the vulnerability of FL, launching attacks with various adversarial attacks or deployment tactics. In this paper, we analyze such vulnerabilities, corresponding solutions were brought forth, and validated through simulation.



## **6. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20361v1) [paper-pdf](http://arxiv.org/pdf/2407.20361v1)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.



## **7. Prompt Leakage effect and defense strategies for multi-turn LLM interactions**

cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2404.16251v3) [paper-pdf](http://arxiv.org/pdf/2404.16251v3)

**Authors**: Divyansh Agarwal, Alexander R. Fabbri, Ben Risher, Philippe Laban, Shafiq Joty, Chien-Sheng Wu

**Abstract**: Prompt leakage poses a compelling security and privacy threat in LLM applications. Leakage of system prompts may compromise intellectual property, and act as adversarial reconnaissance for an attacker. A systematic evaluation of prompt leakage threats and mitigation strategies is lacking, especially for multi-turn LLM interactions. In this paper, we systematically investigate LLM vulnerabilities against prompt leakage for 10 closed- and open-source LLMs, across four domains. We design a unique threat model which leverages the LLM sycophancy effect and elevates the average attack success rate (ASR) from 17.7% to 86.2% in a multi-turn setting. Our standardized setup further allows dissecting leakage of specific prompt contents such as task instructions and knowledge documents. We measure the mitigation effect of 7 black-box defense strategies, along with finetuning an open-source model to defend against leakage attempts. We present different combination of defenses against our threat model, including a cost analysis. Our study highlights key takeaways for building secure LLM applications and provides directions for research in multi-turn LLM interactions



## **8. DDAP: Dual-Domain Anti-Personalization against Text-to-Image Diffusion Models**

cs.CV

Accepted by IJCB 2024

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20141v1) [paper-pdf](http://arxiv.org/pdf/2407.20141v1)

**Authors**: Jing Yang, Runping Xi, Yingxin Lai, Xun Lin, Zitong Yu

**Abstract**: Diffusion-based personalized visual content generation technologies have achieved significant breakthroughs, allowing for the creation of specific objects by just learning from a few reference photos. However, when misused to fabricate fake news or unsettling content targeting individuals, these technologies could cause considerable societal harm. To address this problem, current methods generate adversarial samples by adversarially maximizing the training loss, thereby disrupting the output of any personalized generation model trained with these samples. However, the existing methods fail to achieve effective defense and maintain stealthiness, as they overlook the intrinsic properties of diffusion models. In this paper, we introduce a novel Dual-Domain Anti-Personalization framework (DDAP). Specifically, we have developed Spatial Perturbation Learning (SPL) by exploiting the fixed and perturbation-sensitive nature of the image encoder in personalized generation. Subsequently, we have designed a Frequency Perturbation Learning (FPL) method that utilizes the characteristics of diffusion models in the frequency domain. The SPL disrupts the overall texture of the generated images, while the FPL focuses on image details. By alternating between these two methods, we construct the DDAP framework, effectively harnessing the strengths of both domains. To further enhance the visual quality of the adversarial samples, we design a localization module to accurately capture attentive areas while ensuring the effectiveness of the attack and avoiding unnecessary disturbances in the background. Extensive experiments on facial benchmarks have shown that the proposed DDAP enhances the disruption of personalized generation models while also maintaining high quality in adversarial samples, making it more effective in protecting privacy in practical applications.



## **9. Adversarial Robustness in RGB-Skeleton Action Recognition: Leveraging Attention Modality Reweighter**

cs.CV

Accepted by IJCB 2024

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.19981v1) [paper-pdf](http://arxiv.org/pdf/2407.19981v1)

**Authors**: Chao Liu, Xin Liu, Zitong Yu, Yonghong Hou, Huanjing Yue, Jingyu Yang

**Abstract**: Deep neural networks (DNNs) have been applied in many computer vision tasks and achieved state-of-the-art (SOTA) performance. However, misclassification will occur when DNNs predict adversarial examples which are created by adding human-imperceptible adversarial noise to natural examples. This limits the application of DNN in security-critical fields. In order to enhance the robustness of models, previous research has primarily focused on the unimodal domain, such as image recognition and video understanding. Although multi-modal learning has achieved advanced performance in various tasks, such as action recognition, research on the robustness of RGB-skeleton action recognition models is scarce. In this paper, we systematically investigate how to improve the robustness of RGB-skeleton action recognition models. We initially conducted empirical analysis on the robustness of different modalities and observed that the skeleton modality is more robust than the RGB modality. Motivated by this observation, we propose the \formatword{A}ttention-based \formatword{M}odality \formatword{R}eweighter (\formatword{AMR}), which utilizes an attention layer to re-weight the two modalities, enabling the model to learn more robust features. Our AMR is plug-and-play, allowing easy integration with multimodal models. To demonstrate the effectiveness of AMR, we conducted extensive experiments on various datasets. For example, compared to the SOTA methods, AMR exhibits a 43.77\% improvement against PGD20 attacks on the NTU-RGB+D 60 dataset. Furthermore, it effectively balances the differences in robustness between different modalities.



## **10. Quasi-Framelets: Robust Graph Neural Networks via Adaptive Framelet Convolution**

cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2201.04728v2) [paper-pdf](http://arxiv.org/pdf/2201.04728v2)

**Authors**: Mengxi Yang, Dai Shi, Xuebin Zheng, Jie Yin, Junbin Gao

**Abstract**: This paper aims to provide a novel design of a multiscale framelet convolution for spectral graph neural networks (GNNs). While current spectral methods excel in various graph learning tasks, they often lack the flexibility to adapt to noisy, incomplete, or perturbed graph signals, making them fragile in such conditions. Our newly proposed framelet convolution addresses these limitations by decomposing graph data into low-pass and high-pass spectra through a finely-tuned multiscale approach. Our approach directly designs filtering functions within the spectral domain, allowing for precise control over the spectral components. The proposed design excels in filtering out unwanted spectral information and significantly reduces the adverse effects of noisy graph signals. Our approach not only enhances the robustness of GNNs but also preserves crucial graph features and structures. Through extensive experiments on diverse, real-world graph datasets, we demonstrate that our framelet convolution achieves superior performance in node classification tasks. It exhibits remarkable resilience to noisy data and adversarial attacks, highlighting its potential as a robust solution for real-world graph applications. This advancement opens new avenues for more adaptive and reliable spectral GNN architectures.



## **11. Detecting and Understanding Vulnerabilities in Language Models via Mechanistic Interpretability**

cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.19842v1) [paper-pdf](http://arxiv.org/pdf/2407.19842v1)

**Authors**: Jorge García-Carrasco, Alejandro Maté, Juan Trujillo

**Abstract**: Large Language Models (LLMs), characterized by being trained on broad amounts of data in a self-supervised manner, have shown impressive performance across a wide range of tasks. Indeed, their generative abilities have aroused interest on the application of LLMs across a wide range of contexts. However, neural networks in general, and LLMs in particular, are known to be vulnerable to adversarial attacks, where an imperceptible change to the input can mislead the output of the model. This is a serious concern that impedes the use of LLMs on high-stakes applications, such as healthcare, where a wrong prediction can imply serious consequences. Even though there are many efforts on making LLMs more robust to adversarial attacks, there are almost no works that study \emph{how} and \emph{where} these vulnerabilities that make LLMs prone to adversarial attacks happen. Motivated by these facts, we explore how to localize and understand vulnerabilities, and propose a method, based on Mechanistic Interpretability (MI) techniques, to guide this process. Specifically, this method enables us to detect vulnerabilities related to a concrete task by (i) obtaining the subset of the model that is responsible for that task, (ii) generating adversarial samples for that task, and (iii) using MI techniques together with the previous samples to discover and understand the possible vulnerabilities. We showcase our method on a pretrained GPT-2 Small model carrying out the task of predicting 3-letter acronyms to demonstrate its effectiveness on locating and understanding concrete vulnerabilities of the model.



## **12. Understanding Robust Overfitting from the Feature Generalization Perspective**

cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2310.00607v2) [paper-pdf](http://arxiv.org/pdf/2310.00607v2)

**Authors**: Chaojian Yu, Xiaolong Shi, Jun Yu, Bo Han, Tongliang Liu

**Abstract**: Adversarial training (AT) constructs robust neural networks by incorporating adversarial perturbations into natural data. However, it is plagued by the issue of robust overfitting (RO), which severely damages the model's robustness. In this paper, we investigate RO from a novel feature generalization perspective. Specifically, we design factor ablation experiments to assess the respective impacts of natural data and adversarial perturbations on RO, identifying that the inducing factor of RO stems from natural data. Given that the only difference between adversarial and natural training lies in the inclusion of adversarial perturbations, we further hypothesize that adversarial perturbations degrade the generalization of features in natural data and verify this hypothesis through extensive experiments. Based on these findings, we provide a holistic view of RO from the feature generalization perspective and explain various empirical behaviors associated with RO. To examine our feature generalization perspective, we devise two representative methods, attack strength and data augmentation, to prevent the feature generalization degradation during AT. Extensive experiments conducted on benchmark datasets demonstrate that the proposed methods can effectively mitigate RO and enhance adversarial robustness.



## **13. Exploring the Adversarial Robustness of CLIP for AI-generated Image Detection**

cs.CV

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.19553v1) [paper-pdf](http://arxiv.org/pdf/2407.19553v1)

**Authors**: Vincenzo De Rosa, Fabrizio Guillaro, Giovanni Poggi, Davide Cozzolino, Luisa Verdoliva

**Abstract**: In recent years, many forensic detectors have been proposed to detect AI-generated images and prevent their use for malicious purposes. Convolutional neural networks (CNNs) have long been the dominant architecture in this field and have been the subject of intense study. However, recently proposed Transformer-based detectors have been shown to match or even outperform CNN-based detectors, especially in terms of generalization. In this paper, we study the adversarial robustness of AI-generated image detectors, focusing on Contrastive Language-Image Pretraining (CLIP)-based methods that rely on Visual Transformer backbones and comparing their performance with CNN-based methods. We study the robustness to different adversarial attacks under a variety of conditions and analyze both numerical results and frequency-domain patterns. CLIP-based detectors are found to be vulnerable to white-box attacks just like CNN-based detectors. However, attacks do not easily transfer between CNN-based and CLIP-based methods. This is also confirmed by the different distribution of the adversarial noise patterns in the frequency domain. Overall, this analysis provides new insights into the properties of forensic detectors that can help to develop more effective strategies.



## **14. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

cs.LG

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.12068v2) [paper-pdf](http://arxiv.org/pdf/2407.12068v2)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.



## **15. Breaking the Balance of Power: Commitment Attacks on Ethereum's Reward Mechanism**

cs.CR

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.19479v1) [paper-pdf](http://arxiv.org/pdf/2407.19479v1)

**Authors**: Roozbeh Sarenche, Ertem Nusret Tas, Barnabe Monnot, Caspar Schwarz-Schilling, Bart Preneel

**Abstract**: Validators in permissionless, large-scale blockchains (e.g., Ethereum) are typically payoff-maximizing, rational actors. Ethereum relies on in-protocol incentives, like rewards for validators delivering correct and timely votes, to induce honest behavior and secure the blockchain. However, external incentives, such as the block proposer's opportunity to capture maximal extractable value (MEV), may tempt validators to deviate from honest protocol participation.   We show a series of commitment attacks on LMD GHOST, a core part of Ethereum's consensus mechanism. We demonstrate how a single adversarial block proposer can orchestrate long-range chain reorganizations by manipulating Ethereum's reward system for timely votes. These attacks disrupt the intended balance of power between proposers and voters: by leveraging credible threats, the adversarial proposer can coerce voters from previous slots into supporting blocks that conflict with the honest chain, enabling a chain reorganization at no cost to the adversary. In response, we introduce a novel reward mechanism that restores the voters' role as a check against proposer power. Our proposed mitigation is fairer and more decentralized -- not only in the context of these attacks -- but also practical for implementation in Ethereum.



## **16. Predominant Aspects on Security for Quantum Machine Learning: Literature Review**

quant-ph

Accepted at the IEEE International Conference on Quantum Computing  and Engineering (QCE)

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2401.07774v3) [paper-pdf](http://arxiv.org/pdf/2401.07774v3)

**Authors**: Nicola Franco, Alona Sakhnenko, Leon Stolpmann, Daniel Thuerck, Fabian Petsch, Annika Rüll, Jeanette Miriam Lorenz

**Abstract**: Quantum Machine Learning (QML) has emerged as a promising intersection of quantum computing and classical machine learning, anticipated to drive breakthroughs in computational tasks. This paper discusses the question which security concerns and strengths are connected to QML by means of a systematic literature review. We categorize and review the security of QML models, their vulnerabilities inherent to quantum architectures, and the mitigation strategies proposed. The survey reveals that while QML possesses unique strengths, it also introduces novel attack vectors not seen in classical systems. We point out specific risks, such as cross-talk in superconducting systems and forced repeated shuttle operations in ion-trap systems, which threaten QML's reliability. However, approaches like adversarial training, quantum noise exploitation, and quantum differential privacy have shown potential in enhancing QML robustness. Our review discuss the need for continued and rigorous research to ensure the secure deployment of QML in real-world applications. This work serves as a foundational reference for researchers and practitioners aiming to navigate the security aspects of QML.



## **17. A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.13863v3) [paper-pdf](http://arxiv.org/pdf/2407.13863v3)

**Authors**: Yixiang Qiu, Hao Fang, Hongyao Yu, Bin Chen, MeiKang Qiu, Shu-Tao Xia

**Abstract**: Model Inversion (MI) attacks aim to reconstruct privacy-sensitive training data from released models by utilizing output information, raising extensive concerns about the security of Deep Neural Networks (DNNs). Recent advances in generative adversarial networks (GANs) have contributed significantly to the improved performance of MI attacks due to their powerful ability to generate realistic images with high fidelity and appropriate semantics. However, previous MI attacks have solely disclosed private information in the latent space of GAN priors, limiting their semantic extraction and transferability across multiple target models and datasets. To address this challenge, we propose a novel method, Intermediate Features enhanced Generative Model Inversion (IF-GMI), which disassembles the GAN structure and exploits features between intermediate blocks. This allows us to extend the optimization space from latent code to intermediate features with enhanced expressive capabilities. To prevent GAN priors from generating unrealistic images, we apply a L1 ball constraint to the optimization process. Experiments on multiple benchmarks demonstrate that our method significantly outperforms previous approaches and achieves state-of-the-art results under various settings, especially in the out-of-distribution (OOD) scenario. Our code is available at: https://github.com/final-solution/IF-GMI



## **18. EaTVul: ChatGPT-based Evasion Attack Against Software Vulnerability Detection**

cs.CR

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19216v1) [paper-pdf](http://arxiv.org/pdf/2407.19216v1)

**Authors**: Shigang Liu, Di Cao, Junae Kim, Tamas Abraham, Paul Montague, Seyit Camtepe, Jun Zhang, Yang Xiang

**Abstract**: Recently, deep learning has demonstrated promising results in enhancing the accuracy of vulnerability detection and identifying vulnerabilities in software. However, these techniques are still vulnerable to attacks. Adversarial examples can exploit vulnerabilities within deep neural networks, posing a significant threat to system security. This study showcases the susceptibility of deep learning models to adversarial attacks, which can achieve 100% attack success rate (refer to Table 5). The proposed method, EaTVul, encompasses six stages: identification of important samples using support vector machines, identification of important features using the attention mechanism, generation of adversarial data based on these features using ChatGPT, preparation of an adversarial attack pool, selection of seed data using a fuzzy genetic algorithm, and the execution of an evasion attack. Extensive experiments demonstrate the effectiveness of EaTVul, achieving an attack success rate of more than 83% when the snippet size is greater than 2. Furthermore, in most cases with a snippet size of 4, EaTVul achieves a 100% attack success rate. The findings of this research emphasize the necessity of robust defenses against adversarial attacks in software vulnerability detection.



## **19. Debiased Graph Poisoning Attack via Contrastive Surrogate Objective**

cs.LG

9 pages. Proceeding ACM International Conference on Information and  Knowledge Management (CIKM 2024) Proceeding

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19155v1) [paper-pdf](http://arxiv.org/pdf/2407.19155v1)

**Authors**: Kanghoon Yoon, Yeonjun In, Namkyeong Lee, Kibum Kim, Chanyoung Park

**Abstract**: Graph neural networks (GNN) are vulnerable to adversarial attacks, which aim to degrade the performance of GNNs through imperceptible changes on the graph. However, we find that in fact the prevalent meta-gradient-based attacks, which utilizes the gradient of the loss w.r.t the adjacency matrix, are biased towards training nodes. That is, their meta-gradient is determined by a training procedure of the surrogate model, which is solely trained on the training nodes. This bias manifests as an uneven perturbation, connecting two nodes when at least one of them is a labeled node, i.e., training node, while it is unlikely to connect two unlabeled nodes. However, these biased attack approaches are sub-optimal as they do not consider flipping edges between two unlabeled nodes at all. This means that they miss the potential attacked edges between unlabeled nodes that significantly alter the representation of a node. In this paper, we investigate the meta-gradients to uncover the root cause of the uneven perturbations of existing attacks. Based on our analysis, we propose a Meta-gradient-based attack method using contrastive surrogate objective (Metacon), which alleviates the bias in meta-gradient using a new surrogate loss. We conduct extensive experiments to show that Metacon outperforms existing meta gradient-based attack methods through benchmark datasets, while showing that alleviating the bias towards training nodes is effective in attacking the graph structure.



## **20. A Survey of Malware Detection Using Deep Learning**

cs.CR

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19153v1) [paper-pdf](http://arxiv.org/pdf/2407.19153v1)

**Authors**: Ahmed Bensaoud, Jugal Kalita, Mahmoud Bensaoud

**Abstract**: The problem of malicious software (malware) detection and classification is a complex task, and there is no perfect approach. There is still a lot of work to be done. Unlike most other research areas, standard benchmarks are difficult to find for malware detection. This paper aims to investigate recent advances in malware detection on MacOS, Windows, iOS, Android, and Linux using deep learning (DL) by investigating DL in text and image classification, the use of pre-trained and multi-task learning models for malware detection approaches to obtain high accuracy and which the best approach if we have a standard benchmark dataset. We discuss the issues and the challenges in malware detection using DL classifiers by reviewing the effectiveness of these DL classifiers and their inability to explain their decisions and actions to DL developers presenting the need to use Explainable Machine Learning (XAI) or Interpretable Machine Learning (IML) programs. Additionally, we discuss the impact of adversarial attacks on deep learning models, negatively affecting their generalization capabilities and resulting in poor performance on unseen data. We believe there is a need to train and test the effectiveness and efficiency of the current state-of-the-art deep learning models on different malware datasets. We examine eight popular DL approaches on various datasets. This survey will help researchers develop a general understanding of malware recognition using deep learning.



## **21. Accurate and Scalable Detection and Investigation of Cyber Persistence Threats**

cs.CR

16 pages

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18832v1) [paper-pdf](http://arxiv.org/pdf/2407.18832v1)

**Authors**: Qi Liu, Muhammad Shoaib, Mati Ur Rehman, Kaibin Bao, Veit Hagenmeyer, Wajih Ul Hassan

**Abstract**: In Advanced Persistent Threat (APT) attacks, achieving stealthy persistence within target systems is often crucial for an attacker's success. This persistence allows adversaries to maintain prolonged access, often evading detection mechanisms. Recognizing its pivotal role in the APT lifecycle, this paper introduces Cyber Persistence Detector (CPD), a novel system dedicated to detecting cyber persistence through provenance analytics. CPD is founded on the insight that persistent operations typically manifest in two phases: the "persistence setup" and the subsequent "persistence execution". By causally relating these phases, we enhance our ability to detect persistent threats. First, CPD discerns setups signaling an impending persistent threat and then traces processes linked to remote connections to identify persistence execution activities. A key feature of our system is the introduction of pseudo-dependency edges (pseudo-edges), which effectively connect these disjoint phases using data provenance analysis, and expert-guided edges, which enable faster tracing and reduced log size. These edges empower us to detect persistence threats accurately and efficiently. Moreover, we propose a novel alert triage algorithm that further reduces false positives associated with persistence threats. Evaluations conducted on well-known datasets demonstrate that our system reduces the average false positive rate by 93% compared to state-of-the-art methods.



## **22. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

cs.DC

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2404.14250v4) [paper-pdf](http://arxiv.org/pdf/2404.14250v4)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.



## **23. Embodied Laser Attack:Leveraging Scene Priors to Achieve Agent-based Robust Non-contact Attacks**

cs.CV

9 pages, 7 figures, Accepted by ACM MM 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2312.09554v3) [paper-pdf](http://arxiv.org/pdf/2312.09554v3)

**Authors**: Yitong Sun, Yao Huang, Xingxing Wei

**Abstract**: As physical adversarial attacks become extensively applied in unearthing the potential risk of security-critical scenarios, especially in dynamic scenarios, their vulnerability to environmental variations has also been brought to light. The non-robust nature of physical adversarial attack methods brings less-than-stable performance consequently. Although methods such as EOT have enhanced the robustness of traditional contact attacks like adversarial patches, they fall short in practicality and concealment within dynamic environments such as traffic scenarios. Meanwhile, non-contact laser attacks, while offering enhanced adaptability, face constraints due to a limited optimization space for their attributes, rendering EOT less effective. This limitation underscores the necessity for developing a new strategy to augment the robustness of such practices. To address these issues, this paper introduces the Embodied Laser Attack (ELA), a novel framework that leverages the embodied intelligence paradigm of Perception-Decision-Control to dynamically tailor non-contact laser attacks. For the perception module, given the challenge of simulating the victim's view by full-image transformation, ELA has innovatively developed a local perspective transformation network, based on the intrinsic prior knowledge of traffic scenes and enables effective and efficient estimation. For the decision and control module, ELA trains an attack agent with data-driven reinforcement learning instead of adopting time-consuming heuristic algorithms, making it capable of instantaneously determining a valid attack strategy with the perceived information by well-designed rewards, which is then conducted by a controllable laser emitter. Experimentally, we apply our framework to diverse traffic scenarios both in the digital and physical world, verifying the effectiveness of our method under dynamic successive scenes.



## **24. Adversarial Robustification via Text-to-Image Diffusion Models**

cs.CV

Code is available at https://github.com/ChoiDae1/robustify-T2I

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18658v1) [paper-pdf](http://arxiv.org/pdf/2407.18658v1)

**Authors**: Daewon Choi, Jongheon Jeong, Huiwon Jang, Jinwoo Shin

**Abstract**: Adversarial robustness has been conventionally believed as a challenging property to encode for neural networks, requiring plenty of training data. In the recent paradigm of adopting off-the-shelf models, however, access to their training data is often infeasible or not practical, while most of such models are not originally trained concerning adversarial robustness. In this paper, we develop a scalable and model-agnostic solution to achieve adversarial robustness without using any data. Our intuition is to view recent text-to-image diffusion models as "adaptable" denoisers that can be optimized to specify target tasks. Based on this, we propose: (a) to initiate a denoise-and-classify pipeline that offers provable guarantees against adversarial attacks, and (b) to leverage a few synthetic reference images generated from the text-to-image model that enables novel adaptation schemes. Our experiments show that our data-free scheme applied to the pre-trained CLIP could improve the (provable) adversarial robustness of its diverse zero-shot classification derivatives (while maintaining their accuracy), significantly surpassing prior approaches that utilize the full training data. Not only for CLIP, we also demonstrate that our framework is easily applicable for robustifying other visual classifiers efficiently.



## **25. Robust VAEs via Generating Process of Noise Augmented Data**

cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18632v1) [paper-pdf](http://arxiv.org/pdf/2407.18632v1)

**Authors**: Hiroo Irobe, Wataru Aoki, Kimihiro Yamazaki, Yuhui Zhang, Takumi Nakagawa, Hiroki Waida, Yuichiro Wada, Takafumi Kanamori

**Abstract**: Advancing defensive mechanisms against adversarial attacks in generative models is a critical research topic in machine learning. Our study focuses on a specific type of generative models - Variational Auto-Encoders (VAEs). Contrary to common beliefs and existing literature which suggest that noise injection towards training data can make models more robust, our preliminary experiments revealed that naive usage of noise augmentation technique did not substantially improve VAE robustness. In fact, it even degraded the quality of learned representations, making VAEs more susceptible to adversarial perturbations. This paper introduces a novel framework that enhances robustness by regularizing the latent space divergence between original and noise-augmented data. Through incorporating a paired probabilistic prior into the standard variational lower bound, our method significantly boosts defense against adversarial attacks. Our empirical evaluations demonstrate that this approach, termed Robust Augmented Variational Auto-ENcoder (RAVEN), yields superior performance in resisting adversarial inputs on widely-recognized benchmark datasets.



## **26. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2305.17000v6) [paper-pdf](http://arxiv.org/pdf/2305.17000v6)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **27. Unveiling Privacy Vulnerabilities: Investigating the Role of Structure in Graph Data**

cs.LG

In KDD'24; with full appendix

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18564v1) [paper-pdf](http://arxiv.org/pdf/2407.18564v1)

**Authors**: Hanyang Yuan, Jiarong Xu, Cong Wang, Ziqi Yang, Chunping Wang, Keting Yin, Yang Yang

**Abstract**: The public sharing of user information opens the door for adversaries to infer private data, leading to privacy breaches and facilitating malicious activities. While numerous studies have concentrated on privacy leakage via public user attributes, the threats associated with the exposure of user relationships, particularly through network structure, are often neglected. This study aims to fill this critical gap by advancing the understanding and protection against privacy risks emanating from network structure, moving beyond direct connections with neighbors to include the broader implications of indirect network structural patterns. To achieve this, we first investigate the problem of Graph Privacy Leakage via Structure (GPS), and introduce a novel measure, the Generalized Homophily Ratio, to quantify the various mechanisms contributing to privacy breach risks in GPS. Based on this insight, we develop a novel graph private attribute inference attack, which acts as a pivotal tool for evaluating the potential for privacy leakage through network structures under worst-case scenarios. To protect users' private data from such vulnerabilities, we propose a graph data publishing method incorporating a learnable graph sampling technique, effectively transforming the original graph into a privacy-preserving version. Extensive experiments demonstrate that our attack model poses a significant threat to user privacy, and our graph data publishing method successfully achieves the optimal privacy-utility trade-off compared to baselines.



## **28. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

cs.CR

This work has been accepted by CCS 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2310.15469v3) [paper-pdf](http://arxiv.org/pdf/2310.15469v3)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.



## **29. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

cs.CV

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2406.18849v2) [paper-pdf](http://arxiv.org/pdf/2406.18849v2)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.



## **30. Machine Unlearning using a Multi-GAN based Model**

cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18467v1) [paper-pdf](http://arxiv.org/pdf/2407.18467v1)

**Authors**: Amartya Hatua, Trung T. Nguyen, Andrew H. Sung

**Abstract**: This article presents a new machine unlearning approach that utilizes multiple Generative Adversarial Network (GAN) based models. The proposed method comprises two phases: i) data reorganization in which synthetic data using the GAN model is introduced with inverted class labels of the forget datasets, and ii) fine-tuning the pre-trained model. The GAN models consist of two pairs of generators and discriminators. The generator discriminator pairs generate synthetic data for the retain and forget datasets. Then, a pre-trained model is utilized to get the class labels of the synthetic datasets. The class labels of synthetic and original forget datasets are inverted. Finally, all combined datasets are used to fine-tune the pre-trained model to get the unlearned model. We have performed the experiments on the CIFAR-10 dataset and tested the unlearned models using Membership Inference Attacks (MIA). The inverted class labels procedure and synthetically generated data help to acquire valuable information that enables the model to outperform state-of-the-art models and other standard unlearning classifiers.



## **31. Disrupting Diffusion: Token-Level Attention Erasure Attack against Diffusion-based Customization**

cs.CV

Accepted by ACM MM2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2405.20584v2) [paper-pdf](http://arxiv.org/pdf/2405.20584v2)

**Authors**: Yisu Liu, Jinyang An, Wanqian Zhang, Dayan Wu, Jingzi Gu, Zheng Lin, Weiping Wang

**Abstract**: With the development of diffusion-based customization methods like DreamBooth, individuals now have access to train the models that can generate their personalized images. Despite the convenience, malicious users have misused these techniques to create fake images, thereby triggering a privacy security crisis. In light of this, proactive adversarial attacks are proposed to protect users against customization. The adversarial examples are trained to distort the customization model's outputs and thus block the misuse. In this paper, we propose DisDiff (Disrupting Diffusion), a novel adversarial attack method to disrupt the diffusion model outputs. We first delve into the intrinsic image-text relationships, well-known as cross-attention, and empirically find that the subject-identifier token plays an important role in guiding image generation. Thus, we propose the Cross-Attention Erasure module to explicitly "erase" the indicated attention maps and disrupt the text guidance. Besides,we analyze the influence of the sampling process of the diffusion model on Projected Gradient Descent (PGD) attack and introduce a novel Merit Sampling Scheduler to adaptively modulate the perturbation updating amplitude in a step-aware manner. Our DisDiff outperforms the state-of-the-art methods by 12.75% of FDFR scores and 7.25% of ISM scores across two facial benchmarks and two commonly used prompts on average.



## **32. Regret-Optimal Defense Against Stealthy Adversaries: A System Level Approach**

eess.SY

Accepted, IEEE Conference on Decision and Control (CDC), 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18448v1) [paper-pdf](http://arxiv.org/pdf/2407.18448v1)

**Authors**: Hiroyasu Tsukamoto, Joudi Hajar, Soon-Jo Chung, Fred Y. Hadaegh

**Abstract**: Modern control designs in robotics, aerospace, and cyber-physical systems heavily depend on real-world data obtained through the system outputs. In the face of system faults and malicious attacks, however, these outputs can be compromised to misrepresent some portion of the system information that critically affects their secure and trustworthy operation. In this paper, we introduce a novel regret-optimal control framework for designing controllers that render a linear system robust against stealthy attacks, including sensor and actuator attacks, as well as external disturbances. In particular, we establish (a) a convex optimization-based system metric to quantify the regret with the worst-case stealthy attack (the true performance minus the optimal performance in hindsight with the knowledge of the stealthy attack), which improves and adaptively interpolates $\mathcal{H}_2$ and $\mathcal{H}_{\infty}$ norms in the presence of stealthy adversaries, (b) an optimization problem for minimizing the regret of 1 expressed in the system level parameterization, which is useful for its localized and distributed implementation in large-scale systems, and (c) a rank-constrained optimization problem (i.e., optimization with a convex objective subject to convex constraints and rank constraints) equivalent to the optimization problem of (b). Finally, we conduct a numerical simulation which showcases the effectiveness of our approach.



## **33. Human-Interpretable Adversarial Prompt Attack on Large Language Models with Situational Context**

cs.CL

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.14644v2) [paper-pdf](http://arxiv.org/pdf/2407.14644v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on testing the vulnerabilities in Large Language Models (LLMs) using adversarial attacks has primarily focused on nonsensical prompt injections, which are easily detected upon manual or automated review (e.g., via byte entropy). However, the exploration of innocuous human-understandable malicious prompts augmented with adversarial injections remains limited. In this research, we explore converting a nonsensical suffix attack into a sensible prompt via a situation-driven contextual re-writing. This allows us to show suffix conversion without any gradients, using only LLMs to perform the attacks, and thus better understand the scope of possible risks. We combine an independent, meaningful adversarial insertion and situations derived from movies to check if this can trick an LLM. The situations are extracted from the IMDB dataset, and prompts are defined following a few-shot chain-of-thought prompting. Our approach demonstrates that a successful situation-driven attack can be executed on both open-source and proprietary LLMs. We find that across many LLMs, as few as 1 attempt produces an attack and that these attacks transfer between LLMs.



## **34. Sparse vs Contiguous Adversarial Pixel Perturbations in Multimodal Models: An Empirical Analysis**

cs.CV

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.18251v1) [paper-pdf](http://arxiv.org/pdf/2407.18251v1)

**Authors**: Cristian-Alexandru Botocan, Raphael Meier, Ljiljana Dolamic

**Abstract**: Assessing the robustness of multimodal models against adversarial examples is an important aspect for the safety of its users. We craft L0-norm perturbation attacks on the preprocessed input images. We launch them in a black-box setup against four multimodal models and two unimodal DNNs, considering both targeted and untargeted misclassification. Our attacks target less than 0.04% of perturbed image area and integrate different spatial positioning of perturbed pixels: sparse positioning and pixels arranged in different contiguous shapes (row, column, diagonal, and patch). To the best of our knowledge, we are the first to assess the robustness of three state-of-the-art multimodal models (ALIGN, AltCLIP, GroupViT) against different sparse and contiguous pixel distribution perturbations. The obtained results indicate that unimodal DNNs are more robust than multimodal models. Furthermore, models using CNN-based Image Encoder are more vulnerable than models with ViT - for untargeted attacks, we obtain a 99% success rate by perturbing less than 0.02% of the image area.



## **35. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2312.03853v4) [paper-pdf](http://arxiv.org/pdf/2312.03853v4)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Gemini (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.



## **36. RIDA: A Robust Attack Framework on Incomplete Graphs**

cs.LG

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.18170v1) [paper-pdf](http://arxiv.org/pdf/2407.18170v1)

**Authors**: Jianke Yu, Hanchen Wang, Chen Chen, Xiaoyang Wang, Wenjie Zhang, Ying Zhang

**Abstract**: Graph Neural Networks (GNNs) are vital in data science but are increasingly susceptible to adversarial attacks. To help researchers develop more robust GNN models, it's essential to focus on designing strong attack models as foundational benchmarks and guiding references. Among adversarial attacks, gray-box poisoning attacks are noteworthy due to their effectiveness and fewer constraints. These attacks exploit GNNs' need for retraining on updated data, thereby impacting their performance by perturbing these datasets. However, current research overlooks the real-world scenario of incomplete graphs.To address this gap, we introduce the Robust Incomplete Deep Attack Framework (RIDA). It is the first algorithm for robust gray-box poisoning attacks on incomplete graphs. The approach innovatively aggregates distant vertex information and ensures powerful data utilization.Extensive tests against 9 SOTA baselines on 3 real-world datasets demonstrate RIDA's superiority in handling incompleteness and high attack performance on the incomplete graph.



## **37. Understanding the Security Benefits and Overheads of Emerging Industry Solutions to DRAM Read Disturbance**

cs.CR

To appear in DRAMSec 2024

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2406.19094v2) [paper-pdf](http://arxiv.org/pdf/2406.19094v2)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Oğuz Ergin, Onur Mutlu

**Abstract**: We present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC), described in JEDEC DDR5 specification's April 2024 update. Unlike prior state-of-the-art that advises the memory controller to periodically issue refresh management (RFM) commands, which provides the DRAM chip with time to perform refreshes, PRAC introduces a new back-off signal. PRAC's back-off signal propagates from the DRAM chip to the memory controller and forces the memory controller to 1) stop serving requests and 2) issue RFM commands. As a result, RFM commands are issued when needed as opposed to periodically, reducing RFM's overheads. We analyze PRAC in four steps. First, we define an adversarial access pattern that represents the worst-case for PRAC's security. Second, we investigate PRAC's configurations and security implications. Our analyses show that PRAC can be configured for secure operation as long as no bitflip occurs before accessing a memory location 10 times. Third, we evaluate the performance impact of PRAC and compare it against prior works using Ramulator 2.0. Our analysis shows that while PRAC incurs less than 13% performance overhead for today's DRAM chips, its performance overheads can reach up to 94% for future DRAM chips that are more vulnerable to read disturbance bitflips. Fourth, we define an availability adversarial access pattern that exacerbates PRAC's performance overhead to perform a memory performance attack, demonstrating that such an adversarial pattern can hog up to 94% of DRAM throughput and degrade system throughput by up to 95%. We discuss PRAC's implications on future systems and foreshadow future research directions. To aid future research, we open-source our implementations and scripts at https://github.com/CMU-SAFARI/ramulator2.



## **38. Chernoff Information as a Privacy Constraint for Adversarial Classification**

cs.IT

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2403.10307v2) [paper-pdf](http://arxiv.org/pdf/2403.10307v2)

**Authors**: Ayşe Ünsal, Melek Önen

**Abstract**: This work inspects a privacy metric based on Chernoff information, \textit{Chernoff differential privacy}, due to its significance in characterization of the optimal classifier's performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we focus on the Bayesian setting and characterize the relationship between the best error exponent of the average error probability and $\varepsilon\textrm{-}$differential privacy \cite{D06}. Accordingly, we re-derive Chernoff differential privacy in terms of $\varepsilon\textrm{-}$differential privacy using the Radon-Nikodym derivative and show that it satisfies the composition property for sequential composition. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$, the impact of the adversary's attack and global sensitivity for the problem of adversarial classification in Laplace mechanisms.



## **39. Is the Digital Forensics and Incident Response Pipeline Ready for Text-Based Threats in LLM Era?**

cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17870v1) [paper-pdf](http://arxiv.org/pdf/2407.17870v1)

**Authors**: Avanti Bhandarkar, Ronald Wilson, Anushka Swarup, Mengdi Zhu, Damon Woodard

**Abstract**: In the era of generative AI, the widespread adoption of Neural Text Generators (NTGs) presents new cybersecurity challenges, particularly within the realms of Digital Forensics and Incident Response (DFIR). These challenges primarily involve the detection and attribution of sources behind advanced attacks like spearphishing and disinformation campaigns. As NTGs evolve, the task of distinguishing between human and NTG-authored texts becomes critically complex. This paper rigorously evaluates the DFIR pipeline tailored for text-based security systems, specifically focusing on the challenges of detecting and attributing authorship of NTG-authored texts. By introducing a novel human-NTG co-authorship text attack, termed CS-ACT, our study uncovers significant vulnerabilities in traditional DFIR methodologies, highlighting discrepancies between ideal scenarios and real-world conditions. Utilizing 14 diverse datasets and 43 unique NTGs, up to the latest GPT-4, our research identifies substantial vulnerabilities in the forensic profiling phase, particularly in attributing authorship to NTGs. Our comprehensive evaluation points to factors such as model sophistication and the lack of distinctive style within NTGs as significant contributors for these vulnerabilities. Our findings underscore the necessity for more sophisticated and adaptable strategies, such as incorporating adversarial learning, stylizing NTGs, and implementing hierarchical attribution through the mapping of NTG lineages to enhance source attribution. This sets the stage for future research and the development of more resilient text-based security systems.



## **40. Domain Generalized Recaptured Screen Image Identification Using SWIN Transformer**

cs.CV

11 pages, 10 figures, 9 tables

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17170v2) [paper-pdf](http://arxiv.org/pdf/2407.17170v2)

**Authors**: Preeti Mehta, Aman Sagar, Suchi Kumari

**Abstract**: An increasing number of classification approaches have been developed to address the issue of image rebroadcast and recapturing, a standard attack strategy in insurance frauds, face spoofing, and video piracy. However, most of them neglected scale variations and domain generalization scenarios, performing poorly in instances involving domain shifts, typically made worse by inter-domain and cross-domain scale variances. To overcome these issues, we propose a cascaded data augmentation and SWIN transformer domain generalization framework (DAST-DG) in the current research work Initially, we examine the disparity in dataset representation. A feature generator is trained to make authentic images from various domains indistinguishable. This process is then applied to recaptured images, creating a dual adversarial learning setup. Extensive experiments demonstrate that our approach is practical and surpasses state-of-the-art methods across different databases. Our model achieves an accuracy of approximately 82\% with a precision of 95\% on high-variance datasets.



## **41. A Unified Understanding of Adversarial Vulnerability Regarding Unimodal Models and Vision-Language Pre-training Models**

cs.CV

14 pages, 9 figures, published in ACMMM2024(oral)

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17797v1) [paper-pdf](http://arxiv.org/pdf/2407.17797v1)

**Authors**: Haonan Zheng, Xinyang Deng, Wen Jiang, Wenrui Li

**Abstract**: With Vision-Language Pre-training (VLP) models demonstrating powerful multimodal interaction capabilities, the application scenarios of neural networks are no longer confined to unimodal domains but have expanded to more complex multimodal V+L downstream tasks. The security vulnerabilities of unimodal models have been extensively examined, whereas those of VLP models remain challenging. We note that in CV models, the understanding of images comes from annotated information, while VLP models are designed to learn image representations directly from raw text. Motivated by this discrepancy, we developed the Feature Guidance Attack (FGA), a novel method that uses text representations to direct the perturbation of clean images, resulting in the generation of adversarial images. FGA is orthogonal to many advanced attack strategies in the unimodal domain, facilitating the direct application of rich research findings from the unimodal to the multimodal scenario. By appropriately introducing text attack into FGA, we construct Feature Guidance with Text Attack (FGA-T). Through the interaction of attacking two modalities, FGA-T achieves superior attack effects against VLP models. Moreover, incorporating data augmentation and momentum mechanisms significantly improves the black-box transferability of FGA-T. Our method demonstrates stable and effective attack capabilities across various datasets, downstream tasks, and both black-box and white-box settings, offering a unified baseline for exploring the robustness of VLP models.



## **42. Exploring Semantic Perturbations on Grover**

cs.LG

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2302.00509v2) [paper-pdf](http://arxiv.org/pdf/2302.00509v2)

**Authors**: Ziqing Ji, Pranav Kulkarni, Marko Neskovic, Kevin Nolan, Yan Xu

**Abstract**: With news and information being as easy to access as they currently are, it is more important than ever to ensure that people are not mislead by what they read. Recently, the rise of neural fake news (AI-generated fake news) and its demonstrated effectiveness at fooling humans has prompted the development of models to detect it. One such model is the Grover model, which can both detect neural fake news to prevent it, and generate it to demonstrate how a model could be misused to fool human readers. In this work we explore the Grover model's fake news detection capabilities by performing targeted attacks through perturbations on input news articles. Through this we test Grover's resilience to these adversarial attacks and expose some potential vulnerabilities which should be addressed in further iterations to ensure it can detect all types of fake news accurately.



## **43. Explaining the Model, Protecting Your Data: Revealing and Mitigating the Data Privacy Risks of Post-Hoc Model Explanations via Membership Inference**

cs.CR

ICML 2024 Workshop on the Next Generation of AI Safety

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17663v1) [paper-pdf](http://arxiv.org/pdf/2407.17663v1)

**Authors**: Catherine Huang, Martin Pawelczyk, Himabindu Lakkaraju

**Abstract**: Predictive machine learning models are becoming increasingly deployed in high-stakes contexts involving sensitive personal data; in these contexts, there is a trade-off between model explainability and data privacy. In this work, we push the boundaries of this trade-off: with a focus on foundation models for image classification fine-tuning, we reveal unforeseen privacy risks of post-hoc model explanations and subsequently offer mitigation strategies for such risks. First, we construct VAR-LRT and L1/L2-LRT, two new membership inference attacks based on feature attribution explanations that are significantly more successful than existing explanation-leveraging attacks, particularly in the low false-positive rate regime that allows an adversary to identify specific training set members with confidence. Second, we find empirically that optimized differentially private fine-tuning substantially diminishes the success of the aforementioned attacks, while maintaining high model accuracy. We carry out a systematic empirical investigation of our 2 new attacks with 5 vision transformer architectures, 5 benchmark datasets, 4 state-of-the-art post-hoc explanation methods, and 4 privacy strength settings.



## **44. Revising the Problem of Partial Labels from the Perspective of CNNs' Robustness**

cs.CV

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17630v1) [paper-pdf](http://arxiv.org/pdf/2407.17630v1)

**Authors**: Xin Zhang, Yuqi Song, Wyatt McCurdy, Xiaofeng Wang, Fei Zuo

**Abstract**: Convolutional neural networks (CNNs) have gained increasing popularity and versatility in recent decades, finding applications in diverse domains. These remarkable achievements are greatly attributed to the support of extensive datasets with precise labels. However, annotating image datasets is intricate and complex, particularly in the case of multi-label datasets. Hence, the concept of partial-label setting has been proposed to reduce annotation costs, and numerous corresponding solutions have been introduced. The evaluation methods for these existing solutions have been primarily based on accuracy. That is, their performance is assessed by their predictive accuracy on the test set. However, we insist that such an evaluation is insufficient and one-sided. On one hand, since the quality of the test set has not been evaluated, the assessment results are unreliable. On the other hand, the partial-label problem may also be raised by undergoing adversarial attacks. Therefore, incorporating robustness into the evaluation system is crucial. For this purpose, we first propose two attack models to generate multiple partial-label datasets with varying degrees of label missing rates. Subsequently, we introduce a lightweight partial-label solution using pseudo-labeling techniques and a designed loss function. Then, we employ D-Score to analyze both the proposed and existing methods to determine whether they can enhance robustness while improving accuracy. Extensive experimental results demonstrate that while certain methods may improve accuracy, the enhancement in robustness is not significant, and in some cases, it even diminishes.



## **45. Fluent Student-Teacher Redteaming**

cs.CL

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17447v1) [paper-pdf](http://arxiv.org/pdf/2407.17447v1)

**Authors**: T. Ben Thompson, Michael Sklar

**Abstract**: Many publicly available language models have been safety tuned to reduce the likelihood of toxic or liability-inducing text. Users or security analysts attempt to jailbreak or redteam these models with adversarial prompts which cause compliance with requests. One attack method is to apply discrete optimization techniques to the prompt. However, the resulting attack strings are often gibberish text, easily filtered by defenders due to high measured perplexity, and may fail for unseen tasks and/or well-tuned models. In this work, we improve existing algorithms (primarily GCG and BEAST) to develop powerful and fluent attacks on safety-tuned models like Llama-2 and Phi-3. Our technique centers around a new distillation-based approach that encourages the victim model to emulate a toxified finetune, either in terms of output probabilities or internal activations. To encourage human-fluent attacks, we add a multi-model perplexity penalty and a repetition penalty to the objective. We also enhance optimizer strength by allowing token insertions, token swaps, and token deletions and by using longer attack sequences. The resulting process is able to reliably jailbreak the most difficult target models with prompts that appear similar to human-written prompts. On Advbench we achieve attack success rates $>93$% for Llama-2-7B, Llama-3-8B, and Vicuna-7B, while maintaining model-measured perplexity $<33$; we achieve $95$% attack success for Phi-3, though with higher perplexity. We also find a universally-optimized single fluent prompt that induces $>88$% compliance on previously unseen tasks across Llama-2-7B, Phi-3-mini and Vicuna-7B and transfers to other black-box models.



## **46. Physical Adversarial Attack on Monocular Depth Estimation via Shape-Varying Patches**

cs.CV

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17312v1) [paper-pdf](http://arxiv.org/pdf/2407.17312v1)

**Authors**: Chenxing Zhao, Yang Li, Shihao Wu, Wenyi Tan, Shuangju Zhou, Quan Pan

**Abstract**: Adversarial attacks against monocular depth estimation (MDE) systems pose significant challenges, particularly in safety-critical applications such as autonomous driving. Existing patch-based adversarial attacks for MDE are confined to the vicinity of the patch, making it difficult to affect the entire target. To address this limitation, we propose a physics-based adversarial attack on monocular depth estimation, employing a framework called Attack with Shape-Varying Patches (ASP), aiming to optimize patch content, shape, and position to maximize effectiveness. We introduce various mask shapes, including quadrilateral, rectangular, and circular masks, to enhance the flexibility and efficiency of the attack. Furthermore, we propose a new loss function to extend the influence of the patch beyond the overlapping regions. Experimental results demonstrate that our attack method generates an average depth error of 18 meters on the target car with a patch area of 1/9, affecting over 98\% of the target area.



## **47. Learning to Transform Dynamically for Better Adversarial Transferability**

cs.CV

accepted as a poster in CVPR 2024

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2405.14077v2) [paper-pdf](http://arxiv.org/pdf/2405.14077v2)

**Authors**: Rongyi Zhu, Zeliang Zhang, Susan Liang, Zhuo Liu, Chenliang Xu

**Abstract**: Adversarial examples, crafted by adding perturbations imperceptible to humans, can deceive neural networks. Recent studies identify the adversarial transferability across various models, \textit{i.e.}, the cross-model attack ability of adversarial samples. To enhance such adversarial transferability, existing input transformation-based methods diversify input data with transformation augmentation. However, their effectiveness is limited by the finite number of available transformations. In our study, we introduce a novel approach named Learning to Transform (L2T). L2T increases the diversity of transformed images by selecting the optimal combination of operations from a pool of candidates, consequently improving adversarial transferability. We conceptualize the selection of optimal transformation combinations as a trajectory optimization problem and employ a reinforcement learning strategy to effectively solve the problem. Comprehensive experiments on the ImageNet dataset, as well as practical tests with Google Vision and GPT-4V, reveal that L2T surpasses current methodologies in enhancing adversarial transferability, thereby confirming its effectiveness and practical significance. The code is available at https://github.com/RongyiZhu/L2T.



## **48. When AI Defeats Password Deception! A Deep Learning Framework to Distinguish Passwords and Honeywords**

cs.CR

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.16964v1) [paper-pdf](http://arxiv.org/pdf/2407.16964v1)

**Authors**: Jimmy Dani, Brandon McCulloh, Nitesh Saxena

**Abstract**: "Honeywords" have emerged as a promising defense mechanism for detecting data breaches and foiling offline dictionary attacks (ODA) by deceiving attackers with false passwords. In this paper, we propose PassFilter, a novel deep learning (DL) based attack framework, fundamental in its ability to identify passwords from a set of sweetwords associated with a user account, effectively challenging a variety of honeywords generation techniques (HGTs). The DL model in PassFilter is trained with a set of previously collected or adversarially generated passwords and honeywords, and carefully orchestrated to predict whether a sweetword is the password or a honeyword. Our model can compromise the security of state-of-the-art, heuristics-based, and representation learning-based HGTs proposed by Dionysiou et al. Specifically, our analysis with nine publicly available password datasets shows that PassFilter significantly outperforms the baseline random guessing success rate of 5%, achieving 6.10% to 52.78% on the 1st guessing attempt, considering 20 sweetwords per account. This success rate rapidly increases with additional login attempts before account lock-outs, often allowed on many real-world online services to maintain reasonable usability. For example, it ranges from 41.78% to 96.80% for five attempts, and from 72.87% to 99.00% for ten attempts, compared to 25% and 50% random guessing, respectively. We also examined PassFilter against general-purpose language models used for honeyword generation, like those proposed by Yu et al. These honeywords also proved vulnerable to our attack, with success rates of 14.19% for 1st guessing attempt, increasing to 30.23%, 41.70%, and 63.10% after 3rd, 5th, and 10th guessing attempts, respectively. Our findings demonstrate the effectiveness of DL model deployed in PassFilter in breaching state-of-the-art HGTs and compromising password security based on ODA.



## **49. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2403.13031v2) [paper-pdf](http://arxiv.org/pdf/2403.13031v2)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.



## **50. RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent**

cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16667v1) [paper-pdf](http://arxiv.org/pdf/2407.16667v1)

**Authors**: Huiyu Xu, Wenhui Zhang, Zhibo Wang, Feng Xiao, Rui Zheng, Yunhe Feng, Zhongjie Ba, Kui Ren

**Abstract**: Recently, advanced Large Language Models (LLMs) such as GPT-4 have been integrated into many real-world applications like Code Copilot. These applications have significantly expanded the attack surface of LLMs, exposing them to a variety of threats. Among them, jailbreak attacks that induce toxic responses through jailbreak prompts have raised critical safety concerns. To identify these threats, a growing number of red teaming approaches simulate potential adversarial scenarios by crafting jailbreak prompts to test the target LLM. However, existing red teaming methods do not consider the unique vulnerabilities of LLM in different scenarios, making it difficult to adjust the jailbreak prompts to find context-specific vulnerabilities. Meanwhile, these methods are limited to refining jailbreak templates using a few mutation operations, lacking the automation and scalability to adapt to different scenarios. To enable context-aware and efficient red teaming, we abstract and model existing attacks into a coherent concept called "jailbreak strategy" and propose a multi-agent LLM system named RedAgent that leverages these strategies to generate context-aware jailbreak prompts. By self-reflecting on contextual feedback in an additional memory buffer, RedAgent continuously learns how to leverage these strategies to achieve effective jailbreaks in specific contexts. Extensive experiments demonstrate that our system can jailbreak most black-box LLMs in just five queries, improving the efficiency of existing red teaming methods by two times. Additionally, RedAgent can jailbreak customized LLM applications more efficiently. By generating context-aware jailbreak prompts towards applications on GPTs, we discover 60 severe vulnerabilities of these real-world applications with only two queries per vulnerability. We have reported all found issues and communicated with OpenAI and Meta for bug fixes.



