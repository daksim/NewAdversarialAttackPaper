# Latest Adversarial Attack Papers
**update at 2025-02-19 09:55:41**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Quantum Byzantine Multiple Access Channels**

cs.IT

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.12047v1) [paper-pdf](http://arxiv.org/pdf/2502.12047v1)

**Authors**: Minglai Cai, Christian Deppe

**Abstract**: In communication theory, attacks like eavesdropping or jamming are typically assumed to occur at the channel level, while communication parties are expected to follow established protocols. But what happens if one of the parties turns malicious? In this work, we investigate a compelling scenario: a multiple-access channel with two transmitters and one receiver, where one transmitter deviates from the protocol and acts dishonestly. To address this challenge, we introduce the Byzantine multiple-access classical-quantum channel and derive an achievable communication rate for this adversarial setting.



## **2. FedEAT: A Robustness Optimization Framework for Federated LLMs**

cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11863v1) [paper-pdf](http://arxiv.org/pdf/2502.11863v1)

**Authors**: Yahao Pang, Xingyuan Wu, Xiaojin Zhang, Wei Chen, Hai Jin

**Abstract**: Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creation. However, they still face persistent problems, including substantial computational costs and inadequate availability of training data. The combination of Federated Learning (FL) and LLMs (federated LLMs) offers a solution by leveraging distributed data while protecting privacy, which positions it as an ideal choice for sensitive domains. However, Federated LLMs still suffer from robustness challenges, including data heterogeneity, malicious clients, and adversarial attacks, which greatly hinder their applications. We first introduce the robustness problems in federated LLMs, to address these challenges, we propose FedEAT (Federated Embedding space Adversarial Training), a novel framework that applies adversarial training in the embedding space of client LLM and employs a robust aggregation approach, specifically geometric median aggregation, to enhance the robustness of Federated LLMs. Our experiments demonstrate that FedEAT effectively improves the robustness of Federated LLMs with minimal performance loss.



## **3. Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives**

cs.SD

Accepted by ICLR 2025

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11858v1) [paper-pdf](http://arxiv.org/pdf/2502.11858v1)

**Authors**: Zeliang Zhang, Susan Liang, Daiki Shimada, Chenliang Xu

**Abstract**: While audio-visual learning equips models with a richer understanding of the real world by leveraging multiple sensory modalities, this integration also introduces new vulnerabilities to adversarial attacks.   In this paper, we present a comprehensive study of the adversarial robustness of audio-visual models, considering both temporal and modality-specific vulnerabilities. We propose two powerful adversarial attacks: 1) a temporal invariance attack that exploits the inherent temporal redundancy across consecutive time segments and 2) a modality misalignment attack that introduces incongruence between the audio and visual modalities. These attacks are designed to thoroughly assess the robustness of audio-visual models against diverse threats. Furthermore, to defend against such attacks, we introduce a novel audio-visual adversarial training framework. This framework addresses key challenges in vanilla adversarial training by incorporating efficient adversarial perturbation crafting tailored to multi-modal data and an adversarial curriculum strategy. Extensive experiments in the Kinetics-Sounds dataset demonstrate that our proposed temporal and modality-based attacks in degrading model performance can achieve state-of-the-art performance, while our adversarial training defense largely improves the adversarial robustness as well as the adversarial training efficiency.



## **4. Practical No-box Adversarial Attacks with Training-free Hybrid Image Transformation**

cs.CV

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2203.04607v3) [paper-pdf](http://arxiv.org/pdf/2203.04607v3)

**Authors**: Qilong Zhang, Youheng Sun, Chaoning Zhang, Chaoqun Li, Xuanhan Wang, Jingkuan Song, Lianli Gao

**Abstract**: In recent years, the adversarial vulnerability of deep neural networks (DNNs) has raised increasing attention. Among all the threat models, no-box attacks are the most practical but extremely challenging since they neither rely on any knowledge of the target model or similar substitute model, nor access the dataset for training a new substitute model. Although a recent method has attempted such an attack in a loose sense, its performance is not good enough and computational overhead of training is expensive. In this paper, we move a step forward and show the existence of a \textbf{training-free} adversarial perturbation under the no-box threat model, which can be successfully used to attack different DNNs in real-time. Motivated by our observation that high-frequency component (HFC) domains in low-level features and plays a crucial role in classification, we attack an image mainly by manipulating its frequency components. Specifically, the perturbation is manipulated by suppression of the original HFC and adding of noisy HFC. We empirically and experimentally analyze the requirements of effective noisy HFC and show that it should be regionally homogeneous, repeating and dense. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed no-box method. It attacks ten well-known models with a success rate of \textbf{98.13\%} on average, which outperforms state-of-the-art no-box attacks by \textbf{29.39\%}. Furthermore, our method is even competitive to mainstream transfer-based black-box attacks.



## **5. Federated Multi-Armed Bandits Under Byzantine Attacks**

cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2205.04134v3) [paper-pdf](http://arxiv.org/pdf/2205.04134v3)

**Authors**: Artun Saday, İlker Demirel, Yiğit Yıldırım, Cem Tekin

**Abstract**: Multi-armed bandits (MAB) is a sequential decision-making model in which the learner controls the trade-off between exploration and exploitation to maximize its cumulative reward. Federated multi-armed bandits (FMAB) is an emerging framework where a cohort of learners with heterogeneous local models play an MAB game and communicate their aggregated feedback to a server to learn a globally optimal arm. Two key hurdles in FMAB are communication-efficient learning and resilience to adversarial attacks. To address these issues, we study the FMAB problem in the presence of Byzantine clients who can send false model updates threatening the learning process. We analyze the sample complexity and the regret of $\beta$-optimal arm identification. We borrow tools from robust statistics and propose a median-of-means (MoM)-based online algorithm, Fed-MoM-UCB, to cope with Byzantine clients. In particular, we show that if the Byzantine clients constitute less than half of the cohort, the cumulative regret with respect to $\beta$-optimal arms is bounded over time with high probability, showcasing both communication efficiency and Byzantine resilience. We analyze the interplay between the algorithm parameters, a discernibility margin, regret, communication cost, and the arms' suboptimality gaps. We demonstrate Fed-MoM-UCB's effectiveness against the baselines in the presence of Byzantine attacks via experiments.



## **6. Adversarially Robust CLIP Models Can Induce Better (Robust) Perceptual Metrics**

cs.CV

This work has been accepted for publication in the IEEE Conference on  Secure and Trustworthy Machine Learning (SaTML). The final version will be  available on IEEE Xplore

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11725v1) [paper-pdf](http://arxiv.org/pdf/2502.11725v1)

**Authors**: Francesco Croce, Christian Schlarmann, Naman Deep Singh, Matthias Hein

**Abstract**: Measuring perceptual similarity is a key tool in computer vision. In recent years perceptual metrics based on features extracted from neural networks with large and diverse training sets, e.g. CLIP, have become popular. At the same time, the metrics extracted from features of neural networks are not adversarially robust. In this paper we show that adversarially robust CLIP models, called R-CLIP$_\textrm{F}$, obtained by unsupervised adversarial fine-tuning induce a better and adversarially robust perceptual metric that outperforms existing metrics in a zero-shot setting, and further matches the performance of state-of-the-art metrics while being robust after fine-tuning. Moreover, our perceptual metric achieves strong performance on related tasks such as robust image-to-image retrieval, which becomes especially relevant when applied to "Not Safe for Work" (NSFW) content detection and dataset filtering. While standard perceptual metrics can be easily attacked by a small perturbation completely degrading NSFW detection, our robust perceptual metric maintains high accuracy under an attack while having similar performance for unperturbed images. Finally, perceptual metrics induced by robust CLIP models have higher interpretability: feature inversion can show which images are considered similar, while text inversion can find what images are associated to a given prompt. This also allows us to visualize the very rich visual concepts learned by a CLIP model, including memorized persons, paintings and complex queries.



## **7. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11647v1) [paper-pdf](http://arxiv.org/pdf/2502.11647v1)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.



## **8. Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?**

cs.CL

22 pages, 12 figures, 13 tables

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11598v1) [paper-pdf](http://arxiv.org/pdf/2502.11598v1)

**Authors**: Leyi Pan, Aiwei Liu, Shiyu Huang, Yijian Lu, Xuming Hu, Lijie Wen, Irwin King, Philip S. Yu

**Abstract**: The radioactive nature of Large Language Model (LLM) watermarking enables the detection of watermarks inherited by student models when trained on the outputs of watermarked teacher models, making it a promising tool for preventing unauthorized knowledge distillation. However, the robustness of watermark radioactivity against adversarial actors remains largely unexplored. In this paper, we investigate whether student models can acquire the capabilities of teacher models through knowledge distillation while avoiding watermark inheritance. We propose two categories of watermark removal approaches: pre-distillation removal through untargeted and targeted training data paraphrasing (UP and TP), and post-distillation removal through inference-time watermark neutralization (WN). Extensive experiments across multiple model pairs, watermarking schemes and hyper-parameter settings demonstrate that both TP and WN thoroughly eliminate inherited watermarks, with WN achieving this while maintaining knowledge transfer efficiency and low computational overhead. Given the ongoing deployment of watermarking techniques in production LLMs, these findings emphasize the urgent need for more robust defense strategies. Our code is available at https://github.com/THU-BPM/Watermark-Radioactivity-Attack.



## **9. Adversary-Aware DPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training**

cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11455v1) [paper-pdf](http://arxiv.org/pdf/2502.11455v1)

**Authors**: Fenghua Weng, Jian Lou, Jun Feng, Minlie Huang, Wenjie Wang

**Abstract**: Safety alignment is critical in pre-training large language models (LLMs) to generate responses aligned with human values and refuse harmful queries. Unlike LLM, the current safety alignment of VLMs is often achieved with post-hoc safety fine-tuning. However, these methods are less effective to white-box attacks. To address this, we propose $\textit{Adversary-aware DPO (ADPO)}$, a novel training framework that explicitly considers adversarial. $\textit{Adversary-aware DPO (ADPO)}$ integrates adversarial training into DPO to enhance the safety alignment of VLMs under worst-case adversarial perturbations. $\textit{ADPO}$ introduces two key components: (1) an adversarial-trained reference model that generates human-preferred responses under worst-case perturbations, and (2) an adversarial-aware DPO loss that generates winner-loser pairs accounting for adversarial distortions. By combining these innovations, $\textit{ADPO}$ ensures that VLMs remain robust and reliable even in the presence of sophisticated jailbreak attacks. Extensive experiments demonstrate that $\textit{ADPO}$ outperforms baselines in the safety alignment and general utility of VLMs.



## **10. Dagger Behind Smile: Fool LLMs with a Happy Ending Story**

cs.CL

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2501.13115v2) [paper-pdf](http://arxiv.org/pdf/2501.13115v2)

**Authors**: Xurui Song, Zhixin Xie, Shuo Huai, Jiayi Kong, Jun Luo

**Abstract**: The wide adoption of Large Language Models (LLMs) has attracted significant attention from $\textit{jailbreak}$ attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious contents. However, optimization-based attacks have limited efficiency and transferability, while existing manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to $\textit{positive}$ prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a $\textit{happy ending}$, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request.This has made HEA both efficient and effective, as it requires only up to two turns to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% attack success rate on average. We also provide quantitative explanations for the success of HEA.



## **11. Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System**

cs.AI

15 pages, 11 figures

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11358v1) [paper-pdf](http://arxiv.org/pdf/2502.11358v1)

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Junjie Wang, Yuekai Huang, Zhiyuan Chang, Qing Wang

**Abstract**: Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands through compromised tools, manipulating LLMs to send sensitive information to these tools, which leads to potential privacy breaches. However, existing attack approaches are black-box oriented and rely on static commands that cannot adapt flexibly to the changes in user queries and the invocation chain of tools. It makes malicious commands more likely to be detected by LLM and leads to attack failure. In this paper, we propose AutoCMD, a dynamic attack comment generation approach for information theft attacks in LLM tool-learning systems. Inspired by the concept of mimicking the familiar, AutoCMD is capable of inferring the information utilized by upstream tools in the toolchain through learning on open-source systems and reinforcement with target system examples, thereby generating more targeted commands for information theft. The evaluation results show that AutoCMD outperforms the baselines with +13.2% $ASR_{Theft}$, and can be generalized to new tool-learning systems to expose their information leakage risks. We also design four defense methods to effectively protect tool-learning systems from the attack.



## **12. How to Backdoor Consistency Models?**

cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2410.19785v3) [paper-pdf](http://arxiv.org/pdf/2410.19785v3)

**Authors**: Chengen Wang, Murat Kantarcioglu

**Abstract**: Consistency models are a new class of models that generate images by directly mapping noise to data, allowing for one-step generation and significantly accelerating the sampling process. However, their robustness against adversarial attacks has not yet been thoroughly investigated. In this work, we conduct the first study on the vulnerability of consistency models to backdoor attacks. While previous research has explored backdoor attacks on diffusion models, those studies have primarily focused on conventional diffusion models, employing a customized backdoor training process and objective, whereas consistency models have distinct training processes and objectives. Our proposed framework demonstrates the vulnerability of consistency models to backdoor attacks. During image generation, poisoned consistency models produce images with a Fr\'echet Inception Distance (FID) comparable to that of a clean model when sampling from Gaussian noise. However, once the trigger is activated, they generate backdoor target images. We explore various trigger and target configurations to evaluate the vulnerability of consistency models, including the use of random noise as a trigger. This novel trigger is visually inconspicuous, more challenging to detect, and aligns well with the sampling process of consistency models. Across all configurations, our framework successfully compromises the consistency models while maintaining high utility and specificity. We also examine the stealthiness of our proposed attack, which is attributed to the unique properties of consistency models and the elusive nature of the Gaussian noise trigger. Our code is available at \href{https://github.com/chengenw/backdoorCM}{https://github.com/chengenw/backdoorCM}.



## **13. G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems**

cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.11127v1) [paper-pdf](http://arxiv.org/pdf/2502.11127v1)

**Authors**: Shilong Wang, Guibin Zhang, Miao Yu, Guancheng Wan, Fanci Meng, Chongye Guo, Kun Wang, Yang Wang

**Abstract**: Large Language Model (LLM)-based Multi-agent Systems (MAS) have demonstrated remarkable capabilities in various complex tasks, ranging from collaborative problem-solving to autonomous decision-making. However, as these systems become increasingly integrated into critical applications, their vulnerability to adversarial attacks, misinformation propagation, and unintended behaviors have raised significant concerns. To address this challenge, we introduce G-Safeguard, a topology-guided security lens and treatment for robust LLM-MAS, which leverages graph neural networks to detect anomalies on the multi-agent utterance graph and employ topological intervention for attack remediation. Extensive experiments demonstrate that G-Safeguard: (I) exhibits significant effectiveness under various attack strategies, recovering over 40% of the performance for prompt injection; (II) is highly adaptable to diverse LLM backbones and large-scale MAS; (III) can seamlessly combine with mainstream MAS with security guarantees. The code is available at https://github.com/wslong20/G-safeguard.



## **14. Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning**

cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2004.12571v4) [paper-pdf](http://arxiv.org/pdf/2004.12571v4)

**Authors**: Xinjian Luo, Xianglong Zhang

**Abstract**: Federated learning (FL) is a decentralized model training framework that aims to merge isolated data islands while maintaining data privacy. However, recent studies have revealed that Generative Adversarial Network (GAN) based attacks can be employed in FL to learn the distribution of private datasets and reconstruct recognizable images. In this paper, we exploit defenses against GAN-based attacks in FL and propose a framework, Anti-GAN, to prevent attackers from learning the real distribution of the victim's data. The core idea of Anti-GAN is to manipulate the visual features of private training images to make them indistinguishable to human eyes even restored by attackers. Specifically, Anti-GAN projects the private dataset onto a GAN's generator and combines the generated fake images with the actual images to create the training dataset, which is then used for federated model training. The experimental results demonstrate that Anti-GAN is effective in preventing attackers from learning the distribution of private images while causing minimal harm to the accuracy of the federated model.



## **15. Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction**

cs.CL

21pages, 10 figures

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.11084v1) [paper-pdf](http://arxiv.org/pdf/2502.11084v1)

**Authors**: Yuting Huang, Chengyuan Liu, Yifeng Feng, Chao Wu, Fei Wu, Kun Kuang

**Abstract**: As Large Language Models (LLMs) are widely applied in various domains, the safety of LLMs is increasingly attracting attention to avoid their powerful capabilities being misused. Existing jailbreak methods create a forced instruction-following scenario, or search adversarial prompts with prefix or suffix tokens to achieve a specific representation manually or automatically. However, they suffer from low efficiency and explicit jailbreak patterns, far from the real deployment of mass attacks to LLMs. In this paper, we point out that simply rewriting the original instruction can achieve a jailbreak, and we find that this rewriting approach is learnable and transferable. We propose the Rewrite to Jailbreak (R2J) approach, a transferable black-box jailbreak method to attack LLMs by iteratively exploring the weakness of the LLMs and automatically improving the attacking strategy. The jailbreak is more efficient and hard to identify since no additional features are introduced. Extensive experiments and analysis demonstrate the effectiveness of R2J, and we find that the jailbreak is also transferable to multiple datasets and various types of models with only a few queries. We hope our work motivates further investigation of LLM safety.



## **16. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

cs.CL

We still need to polish our paper

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2412.12145v2) [paper-pdf](http://arxiv.org/pdf/2412.12145v2)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}



## **17. Atoxia: Red-teaming Large Language Models with Target Toxic Answers**

cs.CL

Accepted to Findings of NAACL-2025

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2408.14853v2) [paper-pdf](http://arxiv.org/pdf/2408.14853v2)

**Authors**: Yuhao Du, Zhuo Li, Pengyu Cheng, Xiang Wan, Anningzhe Gao

**Abstract**: Despite the substantial advancements in artificial intelligence, large language models (LLMs) remain being challenged by generation safety. With adversarial jailbreaking prompts, one can effortlessly induce LLMs to output harmful content, causing unexpected negative social impacts. This vulnerability highlights the necessity for robust LLM red-teaming strategies to identify and mitigate such risks before large-scale application. To detect specific types of risks, we propose a novel red-teaming method that $\textbf{A}$ttacks LLMs with $\textbf{T}$arget $\textbf{Toxi}$c $\textbf{A}$nswers ($\textbf{Atoxia}$). Given a particular harmful answer, Atoxia generates a corresponding user query and a misleading answer opening to examine the internal defects of a given LLM. The proposed attacker is trained within a reinforcement learning scheme with the LLM outputting probability of the target answer as the reward. We verify the effectiveness of our method on various red-teaming benchmarks, such as AdvBench and HH-Harmless. The empirical results demonstrate that Atoxia can successfully detect safety risks in not only open-source models but also state-of-the-art black-box models such as GPT-4o.



## **18. JPEG Inspired Deep Learning**

cs.CV

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2410.07081v2) [paper-pdf](http://arxiv.org/pdf/2410.07081v2)

**Authors**: Ahmed H. Salamah, Kaixiang Zheng, Yiwen Liu, En-Hui Yang

**Abstract**: Although it is traditionally believed that lossy image compression, such as JPEG compression, has a negative impact on the performance of deep neural networks (DNNs), it is shown by recent works that well-crafted JPEG compression can actually improve the performance of deep learning (DL). Inspired by this, we propose JPEG-DL, a novel DL framework that prepends any underlying DNN architecture with a trainable JPEG compression layer. To make the quantization operation in JPEG compression trainable, a new differentiable soft quantizer is employed at the JPEG layer, and then the quantization operation and underlying DNN are jointly trained. Extensive experiments show that in comparison with the standard DL, JPEG-DL delivers significant accuracy improvements across various datasets and model architectures while enhancing robustness against adversarial attacks. Particularly, on some fine-grained image classification datasets, JPEG-DL can increase prediction accuracy by as much as 20.9%. Our code is available on https://github.com/AhmedHussKhalifa/JPEG-Inspired-DL.git.



## **19. RoMA: Robust Malware Attribution via Byte-level Adversarial Training with Global Perturbations and Adversarial Consistency Regularization**

cs.CR

11 pages, 4 figures

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2502.07492v2) [paper-pdf](http://arxiv.org/pdf/2502.07492v2)

**Authors**: Yuxia Sun, Huihong Chen, Jingcai Guo, Aoxiang Sun, Zhetao Li, Haolin Liu

**Abstract**: Attributing APT (Advanced Persistent Threat) malware to their respective groups is crucial for threat intelligence and cybersecurity. However, APT adversaries often conceal their identities, rendering attribution inherently adversarial. Existing machine learning-based attribution models, while effective, remain highly vulnerable to adversarial attacks. For example, the state-of-the-art byte-level model MalConv sees its accuracy drop from over 90% to below 2% under PGD (projected gradient descent) attacks. Existing gradient-based adversarial training techniques for malware detection or image processing were applied to malware attribution in this study, revealing that both robustness and training efficiency require significant improvement. To address this, we propose RoMA, a novel single-step adversarial training approach that integrates global perturbations to generate enhanced adversarial samples and employs adversarial consistency regularization to improve representation quality and resilience. A novel APT malware dataset named AMG18, with diverse samples and realistic class imbalances, is introduced for evaluation. Extensive experiments show that RoMA significantly outperforms seven competing methods in both adversarial robustness (e.g., achieving over 80% robust accuracy-more than twice that of the next-best method under PGD attacks) and training efficiency (e.g., more than twice as fast as the second-best method in terms of accuracy), while maintaining superior standard accuracy in non-adversarial scenarios.



## **20. MITRE ATT&CK Applications in Cybersecurity and The Way Forward**

cs.CR

37 pages

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2502.10825v1) [paper-pdf](http://arxiv.org/pdf/2502.10825v1)

**Authors**: Yuning Jiang, Qiaoran Meng, Feiyang Shang, Nay Oo, Le Thi Hong Minh, Hoon Wei Lim, Biplab Sikdar

**Abstract**: The MITRE ATT&CK framework is a widely adopted tool for enhancing cybersecurity, supporting threat intelligence, incident response, attack modeling, and vulnerability prioritization. This paper synthesizes research on its application across these domains by analyzing 417 peer-reviewed publications. We identify commonly used adversarial tactics, techniques, and procedures (TTPs) and examine the integration of natural language processing (NLP) and machine learning (ML) with ATT&CK to improve threat detection and response. Additionally, we explore the interoperability of ATT&CK with other frameworks, such as the Cyber Kill Chain, NIST guidelines, and STRIDE, highlighting its versatility. The paper further evaluates the framework from multiple perspectives, including its effectiveness, validation methods, and sector-specific challenges, particularly in industrial control systems (ICS) and healthcare. We conclude by discussing current limitations and proposing future research directions to enhance the applicability of ATT&CK in dynamic cybersecurity environments.



## **21. Pixel Is Not a Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models**

cs.CV

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2408.11810v3) [paper-pdf](http://arxiv.org/pdf/2408.11810v3)

**Authors**: Chun-Yen Shih, Li-Xuan Peng, Jia-Wei Liao, Ernie Chu, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Diffusion Models have emerged as powerful generative models for high-quality image synthesis, with many subsequent image editing techniques based on them. However, the ease of text-based image editing introduces significant risks, such as malicious editing for scams or intellectual property infringement. Previous works have attempted to safeguard images from diffusion-based editing by adding imperceptible perturbations. These methods are costly and specifically target prevalent Latent Diffusion Models (LDMs), while Pixel-domain Diffusion Models (PDMs) remain largely unexplored and robust against such attacks. Our work addresses this gap by proposing a novel attack framework, AtkPDM. AtkPDM is mainly composed of a feature representation attacking loss that exploits vulnerabilities in denoising UNets and a latent optimization strategy to enhance the naturalness of adversarial images. Extensive experiments demonstrate the effectiveness of our approach in attacking dominant PDM-based editing methods (e.g., SDEdit) while maintaining reasonable fidelity and robustness against common defense methods. Additionally, our framework is extensible to LDMs, achieving comparable performance to existing approaches.



## **22. Robustness-aware Automatic Prompt Optimization**

cs.CL

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2412.18196v2) [paper-pdf](http://arxiv.org/pdf/2412.18196v2)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Hang Gao, Fan Yang, Ruixiang Tang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) depends on the quality of prompts and the semantic and structural integrity of the input data. However, existing prompt generation methods primarily focus on well-structured input data, often neglecting the impact of perturbed inputs on prompt effectiveness. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt does not need access to model parameters and gradients. Instead, BATprompt leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. We evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.



## **23. Learning to Rewrite: Generalized LLM-Generated Text Detection**

cs.CL

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2408.04237v2) [paper-pdf](http://arxiv.org/pdf/2408.04237v2)

**Authors**: Ran Li, Wei Hao, Weiliang Zhao, Junfeng Yang, Chengzhi Mao

**Abstract**: Large language models (LLMs) present significant risks when used to generate non-factual content and spread disinformation at scale. Detecting such LLM-generated content is crucial, yet current detectors often struggle to generalize in open-world contexts. We introduce Learning2Rewrite, a novel framework for detecting AI-generated text with exceptional generalization to unseen domains. Our method leverages the insight that LLMs inherently modify AI-generated content less than human-written text when tasked with rewriting. By training LLMs to minimize alterations on AI-generated inputs, we amplify this disparity, yielding a more distinguishable and generalizable edit distance across diverse text distributions. Extensive experiments on data from 21 independent domains and four major LLMs (GPT-3.5, GPT-4, Gemini, and Llama-3) demonstrate that our detector outperforms state-of-the-art detection methods by up to 23.04% in AUROC for in-distribution tests, 37.26% for out-of-distribution tests, and 48.66% under adversarial attacks. Our unique training objective ensures better generalizability compared to directly training for classification, when leveraging the same amount of parameters. Our findings suggest that reinforcing LLMs' inherent rewriting tendencies offers a robust and scalable solution for detecting AI-generated text.



## **24. Random-Set Neural Networks (RS-NN)**

cs.LG

Published as a conference paper at the Thirteenth International  Conference on Learning Representations (ICLR 2025)

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2307.05772v5) [paper-pdf](http://arxiv.org/pdf/2307.05772v5)

**Authors**: Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, Fabio Cuzzolin

**Abstract**: Machine learning is increasingly deployed in safety-critical domains where erroneous predictions may lead to potentially catastrophic consequences, highlighting the need for learning systems to be aware of how confident they are in their own predictions: in other words, 'to know when they do not know'. In this paper, we propose a novel Random-Set Neural Network (RS-NN) approach to classification which predicts belief functions (rather than classical probability vectors) over the class list using the mathematics of random sets, i.e., distributions over the collection of sets of classes. RS-NN encodes the 'epistemic' uncertainty induced by training sets that are insufficiently representative or limited in size via the size of the convex set of probability vectors associated with a predicted belief function. Our approach outperforms state-of-the-art Bayesian and Ensemble methods in terms of accuracy, uncertainty estimation and out-of-distribution (OoD) detection on multiple benchmarks (CIFAR-10 vs SVHN/Intel-Image, MNIST vs FMNIST/KMNIST, ImageNet vs ImageNet-O). RS-NN also scales up effectively to large-scale architectures (e.g. WideResNet-28-10, VGG16, Inception V3, EfficientNetB2 and ViT-Base-16), exhibits remarkable robustness to adversarial attacks and can provide statistical guarantees in a conformal learning setting.



## **25. VT-GAN: Cooperative Tabular Data Synthesis using Vertical Federated Learning**

cs.LG

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2302.01706v2) [paper-pdf](http://arxiv.org/pdf/2302.01706v2)

**Authors**: Zilong Zhao, Han Wu, Aad Van Moorsel, Lydia Y. Chen

**Abstract**: This paper presents the application of Vertical Federated Learning (VFL) to generate synthetic tabular data using Generative Adversarial Networks (GANs). VFL is a collaborative approach to train machine learning models among distinct tabular data holders, such as financial institutions, who possess disjoint features for the same group of customers. In this paper we introduce the VT-GAN framework, Vertical federated Tabular GAN, and demonstrate that VFL can be successfully used to implement GANs for distributed tabular data in privacy-preserving manner, with performance close to centralized GANs that assume shared data. We make design choices with respect to the distribution of GAN generator and discriminator models and introduce a training-with-shuffling technique so that no party can reconstruct training data from the GAN conditional vector. The paper presents (1) an implementation of VT-GAN, (2) a detailed quality evaluation of the VT-GAN-generated synthetic data, (3) an overall scalability examination of VT-GAN framework, (4) a security analysis on VT-GAN's robustness against Membership Inference Attack with different settings of Differential Privacy, for a range of datasets with diverse distribution characteristics. Our results demonstrate that VT-GAN can consistently generate high-fidelity synthetic tabular data of comparable quality to that generated by a centralized GAN algorithm. The difference in machine learning utility can be as low as 2.7%, even under extremely imbalanced data distributions across clients or with different numbers of clients.



## **26. Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning**

cs.LG

8 pages main, 21 pages appendix with reference. Submitted to ICML  2025

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.02844v2) [paper-pdf](http://arxiv.org/pdf/2502.02844v2)

**Authors**: Sunwoo Lee, Jaebak Hwang, Yonghyeon Jo, Seungyul Han

**Abstract**: Traditional robust methods in multi-agent reinforcement learning (MARL) often struggle against coordinated adversarial attacks in cooperative scenarios. To address this limitation, we propose the Wolfpack Adversarial Attack framework, inspired by wolf hunting strategies, which targets an initial agent and its assisting agents to disrupt cooperation. Additionally, we introduce the Wolfpack-Adversarial Learning for MARL (WALL) framework, which trains robust MARL policies to defend against the proposed Wolfpack attack by fostering system-wide collaboration. Experimental results underscore the devastating impact of the Wolfpack attack and the significant robustness improvements achieved by WALL.



## **27. SWAP Attack: Stealthy Side-Channel Attack on Multi-Tenant Quantum Cloud System**

quant-ph

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.10115v1) [paper-pdf](http://arxiv.org/pdf/2502.10115v1)

**Authors**: Wei Jie Bryan Lee, Siyi Wang, Suman Dutta, Walid El Maouaki, Anupam Chattopadhyay

**Abstract**: The rapid advancement of quantum computing has spurred widespread adoption, with cloud-based quantum devices gaining traction in academia and industry. This shift raises critical concerns about the privacy and security of computations on shared, multi-tenant quantum platforms accessed remotely. Recent studies have shown that crosstalk on shared quantum devices allows adversaries to interfere with victim circuits within a neighborhood. While insightful, these works left unresolved questions regarding the root cause of crosstalk, effective countermeasures, and replicability across circuits. We revisit the crosstalk effect, tracing its origins to the SWAP path between qubits and demonstrating its impact even over long distances. Our results significantly improve the understanding of this phenomenon beyond prior works. The proposed SWAP-based side-channel attack operates in both active and passive modes, as verified on real IBM quantum devices. In the active attack, an attacker executing a single CNOT gate can perturb victim circuits running Grover's Algorithm, reducing expected output accuracy by $81.62\%$ through strategic qubit placement. Moreover, this effect can be modeled to identify qubits more susceptible to attack. The passive attack, leveraging a stealthy circuit as small as $6.25\%$ of the victim's, achieves $100\%$ accuracy in predicting the victim's circuit size when running Simon's Algorithm. These findings challenge the existing defense strategy of maximizing topological distance between circuits, showing that attackers can still extract sensitive information or manipulate results remotely. Our work highlights the urgent need for robust security measures to safeguard quantum computations against emerging threats.



## **28. Fast Proxies for LLM Robustness Evaluation**

cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.10487v1) [paper-pdf](http://arxiv.org/pdf/2502.10487v1)

**Authors**: Tim Beyer, Jan Schuchardt, Leo Schwinn, Stephan Günnemann

**Abstract**: Evaluating the robustness of LLMs to adversarial attacks is crucial for safe deployment, yet current red-teaming methods are often prohibitively expensive. We compare the ability of fast proxy metrics to predict the real-world robustness of an LLM against a simulated attacker ensemble. This allows us to estimate a model's robustness to computationally expensive attacks without requiring runs of the attacks themselves. Specifically, we consider gradient-descent-based embedding-space attacks, prefilling attacks, and direct prompting. Even though direct prompting in particular does not achieve high ASR, we find that it and embedding-space attacks can predict attack success rates well, achieving $r_p=0.87$ (linear) and $r_s=0.94$ (Spearman rank) correlations with the full attack ensemble while reducing computational cost by three orders of magnitude.



## **29. ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech**

eess.AS

Database link: https://zenodo.org/records/14498691, Database mirror  link: https://huggingface.co/datasets/jungjee/asvspoof5, ASVspoof 5 Challenge  Workshop Proceeding: https://www.isca-archive.org/asvspoof_2024/index.html

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.08857v2) [paper-pdf](http://arxiv.org/pdf/2502.08857v2)

**Authors**: Xin Wang, Héctor Delgado, Hemlata Tak, Jee-weon Jung, Hye-jin Shim, Massimiliano Todisco, Ivan Kukanov, Xuechen Liu, Md Sahidullah, Tomi Kinnunen, Nicholas Evans, Kong Aik Lee, Junichi Yamagishi, Myeonghun Jeong, Ge Zhu, Yongyi Zang, You Zhang, Soumi Maiti, Florian Lux, Nicolas Müller, Wangyou Zhang, Chengzhe Sun, Shuwei Hou, Siwei Lyu, Sébastien Le Maguer, Cheng Gong, Hanjie Guo, Liping Chen, Vishwanath Singh

**Abstract**: ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake attacks as well as the design of detection solutions. We introduce the ASVspoof 5 database which is generated in crowdsourced fashion from data collected in diverse acoustic conditions (cf. studio-quality data for earlier ASVspoof databases) and from ~2,000 speakers (cf. ~100 earlier). The database contains attacks generated with 32 different algorithms, also crowdsourced, and optimised to varying degrees using new surrogate detection models. Among them are attacks generated with a mix of legacy and contemporary text-to-speech synthesis and voice conversion models, in addition to adversarial attacks which are incorporated for the first time. ASVspoof 5 protocols comprise seven speaker-disjoint partitions. They include two distinct partitions for the training of different sets of attack models, two more for the development and evaluation of surrogate detection models, and then three additional partitions which comprise the ASVspoof 5 training, development and evaluation sets. An auxiliary set of data collected from an additional 30k speakers can also be used to train speaker encoders for the implementation of attack algorithms. Also described herein is an experimental validation of the new ASVspoof 5 database using a set of automatic speaker verification and spoof/deepfake baseline detectors. With the exception of protocols and tools for the generation of spoofed/deepfake speech, the resources described in this paper, already used by participants of the ASVspoof 5 challenge in 2024, are now all freely available to the community.



## **30. What You See Is Not Always What You Get: An Empirical Study of Code Comprehension by Large Language Models**

cs.SE

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2412.08098v2) [paper-pdf](http://arxiv.org/pdf/2412.08098v2)

**Authors**: Bangshuo Zhu, Jiawen Wen, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering tasks, including code generation and comprehension. While LLMs have shown significant potential in assisting with coding, it is perceived that LLMs are vulnerable to adversarial attacks. In this paper, we investigate the vulnerability of LLMs to imperceptible attacks, where hidden character manipulation in source code misleads LLMs' behaviour while remaining undetectable to human reviewers. We devise these attacks into four distinct categories and analyse their impacts on code analysis and comprehension tasks. These four types of imperceptible coding character attacks include coding reordering, invisible coding characters, code deletions, and code homoglyphs. To comprehensively benchmark the robustness of current LLMs solutions against the attacks, we present a systematic experimental evaluation on multiple state-of-the-art LLMs. Our experimental design introduces two key performance metrics, namely model confidence using log probabilities of response, and the response correctness. A set of controlled experiments are conducted using a large-scale perturbed and unperturbed code snippets as the primary prompt input. Our findings confirm the susceptibility of LLMs to imperceptible coding character attacks, while different LLMs present different negative correlations between perturbation magnitude and performance. These results highlight the urgent need for robust LLMs capable of manoeuvring behaviours under imperceptible adversarial conditions. We anticipate this work provides valuable insights for enhancing the security and trustworthiness of LLMs in software engineering applications.



## **31. Siren Song: Manipulating Pose Estimation in XR Headsets Using Acoustic Attacks**

cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.08865v2) [paper-pdf](http://arxiv.org/pdf/2502.08865v2)

**Authors**: Zijian Huang, Yicheng Zhang, Sophie Chen, Nael Abu-Ghazaleh, Jiasi Chen

**Abstract**: Extended Reality (XR) experiences involve interactions between users, the real world, and virtual content. A key step to enable these experiences is the XR headset sensing and estimating the user's pose in order to accurately place and render virtual content in the real world. XR headsets use multiple sensors (e.g., cameras, inertial measurement unit) to perform pose estimation and improve its robustness, but this provides an attack surface for adversaries to interfere with the pose estimation process. In this paper, we create and study the effects of acoustic attacks that create false signals in the inertial measurement unit (IMU) on XR headsets, leading to adverse downstream effects on XR applications. We generate resonant acoustic signals on a HoloLens 2 and measure the resulting perturbations in the IMU readings, and also demonstrate both fine-grained and coarse attacks on the popular ORB-SLAM3 and an open-source XR system (ILLIXR). With the knowledge gleaned from attacking these open-source frameworks, we demonstrate four end-to-end proof-of-concept attacks on a HoloLens 2: manipulating user input, clickjacking, zone invasion, and denial of user interaction. Our experiments show that current commercial XR headsets are susceptible to acoustic attacks, raising concerns for their security.



## **32. Towards Reliable Empirical Machine Unlearning Evaluation: A Cryptographic Game Perspective**

cs.LG

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2404.11577v3) [paper-pdf](http://arxiv.org/pdf/2404.11577v3)

**Authors**: Yiwen Tu, Pingbang Hu, Jiaqi Ma

**Abstract**: Machine unlearning updates machine learning models to remove information from specific training samples, complying with data protection regulations that allow individuals to request the removal of their personal data. Despite the recent development of numerous unlearning algorithms, reliable evaluation of these algorithms remains an open research question. In this work, we focus on membership inference attack (MIA) based evaluation, one of the most common approaches for evaluating unlearning algorithms, and address various pitfalls of existing evaluation metrics lacking theoretical understanding and reliability. Specifically, by modeling the proposed evaluation process as a \emph{cryptographic game} between unlearning algorithms and MIA adversaries, the naturally-induced evaluation metric measures the data removal efficacy of unlearning algorithms and enjoys provable guarantees that existing evaluation metrics fail to satisfy. Furthermore, we propose a practical and efficient approximation of the induced evaluation metric and demonstrate its effectiveness through both theoretical analysis and empirical experiments. Overall, this work presents a novel and reliable approach to empirically evaluating unlearning algorithms, paving the way for the development of more effective unlearning techniques.



## **33. $\textrm{A}^{\textrm{2}}$RNet: Adversarial Attack Resilient Network for Robust Infrared and Visible Image Fusion**

cs.CV

9 pages, 8 figures, The 39th Annual AAAI Conference on Artificial  Intelligence

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2412.09954v3) [paper-pdf](http://arxiv.org/pdf/2412.09954v3)

**Authors**: Jiawei Li, Hongwei Yu, Jiansheng Chen, Xinlong Ding, Jinlong Wang, Jinyuan Liu, Bochao Zou, Huimin Ma

**Abstract**: Infrared and visible image fusion (IVIF) is a crucial technique for enhancing visual performance by integrating unique information from different modalities into one fused image. Exiting methods pay more attention to conducting fusion with undisturbed data, while overlooking the impact of deliberate interference on the effectiveness of fusion results. To investigate the robustness of fusion models, in this paper, we propose a novel adversarial attack resilient network, called $\textrm{A}^{\textrm{2}}$RNet. Specifically, we develop an adversarial paradigm with an anti-attack loss function to implement adversarial attacks and training. It is constructed based on the intrinsic nature of IVIF and provide a robust foundation for future research advancements. We adopt a Unet as the pipeline with a transformer-based defensive refinement module (DRM) under this paradigm, which guarantees fused image quality in a robust coarse-to-fine manner. Compared to previous works, our method mitigates the adverse effects of adversarial perturbations, consistently maintaining high-fidelity fusion results. Furthermore, the performance of downstream tasks can also be well maintained under adversarial attacks. Code is available at https://github.com/lok-18/A2RNet.



## **34. LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection**

cs.LG

PAKDD 2025

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.09271v2) [paper-pdf](http://arxiv.org/pdf/2502.09271v2)

**Authors**: Wenlun Zhang, Enyan Dai, Kentaro Yoshioka

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.



## **35. SoK: State of the time: On Trustworthiness of Digital Clocks**

cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.09837v1) [paper-pdf](http://arxiv.org/pdf/2502.09837v1)

**Authors**: Adeel Nasrullah, Fatima M. Anwar

**Abstract**: Despite the critical role of timing infrastructure in enabling essential services, from public key infrastructure and smart grids to autonomous navigation and high-frequency trading, modern timing stacks remain highly vulnerable to malicious attacks. These threats emerge due to several reasons, including inadequate security mechanisms, the timing architectures unique vulnerability to delays, and implementation issues. In this paper, we aim to obtain a holistic understanding of the issues that make the timing stacks vulnerable to adversarial manipulations, what the challenges are in securing them, and what solutions can be borrowed from the research community to address them. To this end, we perform a systematic analysis of the security vulnerabilities of the timing stack. In doing so, we discover new attack surfaces, i.e., physical timing components and on-device timekeeping, which are often overlooked by existing research that predominantly studies the security of time synchronization protocols. We also show that the emerging trusted timing architectures are flawed and risk compromising wider system security, and propose an alternative design using hardware-software co-design.



## **36. `Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.00735v2) [paper-pdf](http://arxiv.org/pdf/2502.00735v2)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.



## **37. Enhancing Jailbreak Attacks via Compliance-Refusal-Based Initialization**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09755v1) [paper-pdf](http://arxiv.org/pdf/2502.09755v1)

**Authors**: Amit Levi, Rom Himelstein, Yaniv Nemcovsky, Avi Mendelson, Chaim Baskin

**Abstract**: Jailbreak attacks aim to exploit large language models (LLMs) and pose a significant threat to their proper conduct; they seek to bypass models' safeguards and often provoke transgressive behaviors. However, existing automatic jailbreak attacks require extensive computational resources and are prone to converge on suboptimal solutions. In this work, we propose \textbf{C}ompliance \textbf{R}efusal \textbf{I}nitialization (CRI), a novel, attack-agnostic framework that efficiently initializes the optimization in the proximity of the compliance subspace of harmful prompts. By narrowing the initial gap to the adversarial objective, CRI substantially improves adversarial success rates (ASR) and drastically reduces computational overhead -- often requiring just a single optimization step. We evaluate CRI on the widely-used AdvBench dataset over the standard jailbreak attacks of GCG and AutoDAN. Results show that CRI boosts ASR and decreases the median steps to success by up to \textbf{\(\times 60\)}. The project page, along with the reference implementation, is publicly available at \texttt{https://amit1221levi.github.io/CRI-Jailbreak-Init-LLMs-evaluation/}.



## **38. SyntheticPop: Attacking Speaker Verification Systems With Synthetic VoicePops**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09553v1) [paper-pdf](http://arxiv.org/pdf/2502.09553v1)

**Authors**: Eshaq Jamdar, Amith Kamath Belman

**Abstract**: Voice Authentication (VA), also known as Automatic Speaker Verification (ASV), is a widely adopted authentication method, particularly in automated systems like banking services, where it serves as a secondary layer of user authentication. Despite its popularity, VA systems are vulnerable to various attacks, including replay, impersonation, and the emerging threat of deepfake audio that mimics the voice of legitimate users. To mitigate these risks, several defense mechanisms have been proposed. One such solution, Voice Pops, aims to distinguish an individual's unique phoneme pronunciations during the enrollment process. While promising, the effectiveness of VA+VoicePop against a broader range of attacks, particularly logical or adversarial attacks, remains insufficiently explored. We propose a novel attack method, which we refer to as SyntheticPop, designed to target the phoneme recognition capabilities of the VA+VoicePop system. The SyntheticPop attack involves embedding synthetic "pop" noises into spoofed audio samples, significantly degrading the model's performance. We achieve an attack success rate of over 95% while poisoning 20% of the training dataset. Our experiments demonstrate that VA+VoicePop achieves 69% accuracy under normal conditions, 37% accuracy when subjected to a baseline label flipping attack, and just 14% accuracy under our proposed SyntheticPop attack, emphasizing the effectiveness of our method.



## **39. Bayes-Nash Generative Privacy Against Membership Inference Attacks**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2406.01811

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2410.07414v3) [paper-pdf](http://arxiv.org/pdf/2410.07414v3)

**Authors**: Tao Zhang, Rajagopal Venkatesaraman, Rajat K. De, Bradley A. Malin, Yevgeniy Vorobeychik

**Abstract**: Membership inference attacks (MIAs) expose significant privacy risks by determining whether an individual's data is in a dataset. While differential privacy (DP) mitigates such risks, it has several limitations in achieving an optimal balance between utility and privacy, include limited resolution in expressing this tradeoff in only a few privacy parameters, and intractable sensitivity calculations that may be necessary to provide tight privacy guarantees. We propose a game-theoretic framework that models privacy protection from MIA as a Bayesian game between a defender and an attacker. In this game, a dataset is the defender's private information, with privacy loss to the defender (which is gain to the attacker) captured in terms of the attacker's ability to infer membership of individuals in the dataset. To address the strategic complexity of this game, we represent the mixed strategy of the defender as a neural network generator which maps a private dataset to its public representation (for example, noisy summary statistics), while the mixed strategy of the attacker is captured by a discriminator which makes membership inference claims. We refer to the resulting computational approach as a general-sum Generative Adversarial Network, which is trained iteratively by alternating generator and discriminator updates akin to conventional GANs. We call the defender's data sharing policy thereby obtained Bayes-Nash Generative Privacy (BNGP). The BNGP strategy avoids sensitivity calculations, supports compositions of correlated mechanisms, is robust to the attacker's heterogeneous preferences over true and false positives, and yields provable differential privacy guarantees, albeit in an idealized setting.



## **40. On the Importance of Backbone to the Adversarial Robustness of Object Detectors**

cs.CV

Accepted by IEEE TIFS

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2305.17438v2) [paper-pdf](http://arxiv.org/pdf/2305.17438v2)

**Authors**: Xiao Li, Hang Chen, Xiaolin Hu

**Abstract**: Object detection is a critical component of various security-sensitive applications, such as autonomous driving and video surveillance. However, existing object detectors are vulnerable to adversarial attacks, which poses a significant challenge to their reliability and security. Through experiments, first, we found that existing works on improving the adversarial robustness of object detectors give a false sense of security. Second, we found that adversarially pre-trained backbone networks were essential for enhancing the adversarial robustness of object detectors. We then proposed a simple yet effective recipe for fast adversarial fine-tuning on object detectors with adversarially pre-trained backbones. Without any modifications to the structure of object detectors, our recipe achieved significantly better adversarial robustness than previous works. Finally, we explored the potential of different modern object detector designs for improving adversarial robustness with our recipe and demonstrated interesting findings, which inspired us to design state-of-the-art (SOTA) robust detectors. Our empirical results set a new milestone for adversarially robust object detection. Code and trained checkpoints are available at https://github.com/thu-ml/oddefense.



## **41. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

cs.LG

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2408.00315v3) [paper-pdf](http://arxiv.org/pdf/2408.00315v3)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.



## **42. Wasserstein distributional adversarial training for deep neural networks**

cs.LG

15 pages, 4 figures

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09352v1) [paper-pdf](http://arxiv.org/pdf/2502.09352v1)

**Authors**: Xingjian Bai, Guangyi He, Yifan Jiang, Jan Obloj

**Abstract**: Design of adversarial attacks for deep neural networks, as well as methods of adversarial training against them, are subject of intense research. In this paper, we propose methods to train against distributional attack threats, extending the TRADES method used for pointwise attacks. Our approach leverages recent contributions and relies on sensitivity analysis for Wasserstein distributionally robust optimization problems. We introduce an efficient fine-tuning method which can be deployed on a previously trained model. We test our methods on a range of pre-trained models on RobustBench. These experimental results demonstrate the additional training enhances Wasserstein distributional robustness, while maintaining original levels of pointwise robustness, even for already very successful networks. The improvements are less marked for models pre-trained using huge synthetic datasets of 20-100M images. However, remarkably, sometimes our methods are still able to improve their performance even when trained using only the original training dataset (50k images).



## **43. FLAME: Flexible LLM-Assisted Moderation Engine**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09175v1) [paper-pdf](http://arxiv.org/pdf/2502.09175v1)

**Authors**: Ivan Bakulin, Ilia Kopanichuk, Iaroslav Bespalov, Nikita Radchenko, Vladimir Shaposhnikov, Dmitry Dylov, Ivan Oseledets

**Abstract**: The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs.



## **44. Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks**

cs.CV

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09110v1) [paper-pdf](http://arxiv.org/pdf/2502.09110v1)

**Authors**: Eylon Mizrahi, Raz Lapid, Moshe Sipper

**Abstract**: Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.



## **45. Universal Adversarial Attack on Aligned Multimodal LLMs**

cs.AI

Added an affiliation

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.07987v2) [paper-pdf](http://arxiv.org/pdf/2502.07987v2)

**Authors**: Temurbek Rahmatullaev, Polina Druzhinina, Matvey Mikhalchuk, Andrey Kuznetsov, Anton Razzhigaev

**Abstract**: We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.



## **46. RLSA-PFL: Robust Lightweight Secure Aggregation with Model Inconsistency Detection in Privacy-Preserving Federated Learning**

cs.CR

16 pages, 10 Figures

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.08989v1) [paper-pdf](http://arxiv.org/pdf/2502.08989v1)

**Authors**: Nazatul H. Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba

**Abstract**: Federated Learning (FL) allows users to collaboratively train a global machine learning model by sharing local model only, without exposing their private data to a central server. This distributed learning is particularly appealing in scenarios where data privacy is crucial, and it has garnered substantial attention from both industry and academia. However, studies have revealed privacy vulnerabilities in FL, where adversaries can potentially infer sensitive information from the shared model parameters. In this paper, we present an efficient masking-based secure aggregation scheme utilizing lightweight cryptographic primitives to mitigate privacy risks. Our scheme offers several advantages over existing methods. First, it requires only a single setup phase for the entire FL training session, significantly reducing communication overhead. Second, it minimizes user-side overhead by eliminating the need for user-to-user interactions, utilizing an intermediate server layer and a lightweight key negotiation method. Third, the scheme is highly resilient to user dropouts, and the users can join at any FL round. Fourth, it can detect and defend against malicious server activities, including recently discovered model inconsistency attacks. Finally, our scheme ensures security in both semi-honest and malicious settings. We provide security analysis to formally prove the robustness of our approach. Furthermore, we implemented an end-to-end prototype of our scheme. We conducted comprehensive experiments and comparisons, which show that it outperforms existing solutions in terms of communication and computation overhead, functionality, and security.



## **47. An Engorgio Prompt Makes Large Language Model Babble on**

cs.CR

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2412.19394v2) [paper-pdf](http://arxiv.org/pdf/2412.19394v2)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu, Han Qiu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is released at: https://github.com/jianshuod/Engorgio-prompt.



## **48. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2410.02890v3) [paper-pdf](http://arxiv.org/pdf/2410.02890v3)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.



## **49. Bankrupting DoS Attackers**

cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2205.08287v4) [paper-pdf](http://arxiv.org/pdf/2205.08287v4)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstract**: Can we make a denial-of-service attacker pay more than the server and honest clients? Consider a model where a server sees a stream of jobs sent by either honest clients or an adversary. The server sets a price for servicing each job with the aid of an estimator, which provides approximate statistical information about the distribution of previously occurring good jobs.   We describe and analyze pricing algorithms for the server under different models of synchrony, with total cost parameterized by the accuracy of the estimator. Given a reasonably accurate estimator, the algorithm's cost provably grows more slowly than the attacker's cost, as the attacker's cost grows large. Additionally, we prove a lower bound, showing that our pricing algorithm yields asymptotically tight results when the estimator is accurate within constant factors.



## **50. Extreme vulnerability to intruder attacks destabilizes network dynamics**

nlin.AO

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08552v1) [paper-pdf](http://arxiv.org/pdf/2502.08552v1)

**Authors**: Amirhossein Nazerian, Sahand Tangerami, Malbor Asllani, David Phillips, Hernan Makse, Francesco Sorrentino

**Abstract**: Consensus, synchronization, formation control, and power grid balance are all examples of virtuous dynamical states that may arise in networks. Here we focus on how such states can be destabilized from a fundamental perspective; namely, we address the question of how one or a few intruder agents within an otherwise functioning network may compromise its dynamics. We show that a single adversarial node coupled via adversarial couplings to one or more other nodes is sufficient to destabilize the entire network, which we prove to be more efficient than targeting multiple nodes. Then, we show that concentrating the attack on a single low-indegree node induces the greatest instability, challenging the common assumption that hubs are the most critical nodes. This leads to a new characterization of the vulnerability of a node, which contrasts with previous work, and identifies low-indegree nodes (as opposed to the hubs) as the most vulnerable components of a network. Our results are derived for linear systems but hold true for nonlinear networks, including those described by the Kuramoto model. Finally, we derive scaling laws showing that larger networks are less susceptible, on average, to single-node attacks. Overall, these findings highlight an intrinsic vulnerability of technological systems such as autonomous networks, sensor networks, power grids, and the internet of things, which also extend to the realm of complex social and biological networks.



