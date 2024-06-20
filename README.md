# Latest Adversarial Attack Papers
**update at 2024-06-20 09:40:05**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Can Go AIs be adversarially robust?**

cs.LG

67 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12843v1) [paper-pdf](http://arxiv.org/pdf/2406.12843v1)

**Authors**: Tom Tseng, Euan McLean, Kellin Pelrine, Tony T. Wang, Adam Gleave

**Abstract**: Prior work found that superhuman Go AIs like KataGo can be defeated by simple adversarial strategies. In this paper, we study if simple defenses can improve KataGo's worst-case performance. We test three natural defenses: adversarial training on hand-constructed positions, iterated adversarial training, and changing the network architecture. We find that some of these defenses are able to protect against previously discovered attacks. Unfortunately, we also find that none of these defenses are able to withstand adaptive attacks. In particular, we are able to train new adversaries that reliably defeat our defended agents by causing them to blunder in ways humans would not. Our results suggest that building robust AI systems is challenging even in narrow domains such as Go. For interactive examples of attacks and a link to our codebase, see https://goattack.far.ai.



## **2. Adversarial Attacks on Multimodal Agents**

cs.LG

19 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12814v1) [paper-pdf](http://arxiv.org/pdf/2406.12814v1)

**Authors**: Chen Henry Wu, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan

**Abstract**: Vision-enabled language models (VLMs) are now used to build autonomous multimodal agents capable of taking actions in real environments. In this paper, we show that multimodal agents raise new safety risks, even though attacking agents is more challenging than prior attacks due to limited access to and knowledge about the environment. Our attacks use adversarial text strings to guide gradient-based perturbation over one trigger image in the environment: (1) our captioner attack attacks white-box captioners if they are used to process images into captions as additional inputs to the VLM; (2) our CLIP attack attacks a set of CLIP models jointly, which can transfer to proprietary VLMs. To evaluate the attacks, we curated VisualWebArena-Adv, a set of adversarial tasks based on VisualWebArena, an environment for web-based multimodal agent tasks. Within an L-infinity norm of $16/256$ on a single image, the captioner attack can make a captioner-augmented GPT-4V agent execute the adversarial goals with a 75% success rate. When we remove the captioner or use GPT-4V to generate its own captions, the CLIP attack can achieve success rates of 21% and 43%, respectively. Experiments on agents based on other VLMs, such as Gemini-1.5, Claude-3, and GPT-4o, show interesting differences in their robustness. Further analysis reveals several key factors contributing to the attack's success, and we also discuss the implications for defenses as well. Project page: https://chenwu.io/attack-agent Code and data: https://github.com/ChenWu98/agent-attack



## **3. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

cs.CR

Updates in the v2: more models (Llama3, Phi-3, Nemotron-4-340B),  jailbreak artifacts for all attacks are available, evaluation of  generalization to a different judge (Llama-3-70B and Llama Guard 2), more  experiments (convergence plots over iterations, ablation on the suffix length  for random search), improved exposition of the paper, examples of jailbroken  generation

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2404.02151v2) [paper-pdf](http://arxiv.org/pdf/2404.02151v2)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token ``Sure''), potentially with multiple restarts. In this way, we achieve nearly 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.



## **4. UIFV: Data Reconstruction Attack in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12588v1) [paper-pdf](http://arxiv.org/pdf/2406.12588v1)

**Authors**: Jirui Yang, Peng Chen, Zhihui Lu, Qiang Duan, Yubing Bao

**Abstract**: Vertical Federated Learning (VFL) facilitates collaborative machine learning without the need for participants to share raw private data. However, recent studies have revealed privacy risks where adversaries might reconstruct sensitive features through data leakage during the learning process. Although data reconstruction methods based on gradient or model information are somewhat effective, they reveal limitations in VFL application scenarios. This is because these traditional methods heavily rely on specific model structures and/or have strict limitations on application scenarios. To address this, our study introduces the Unified InverNet Framework into VFL, which yields a novel and flexible approach (dubbed UIFV) that leverages intermediate feature data to reconstruct original data, instead of relying on gradients or model details. The intermediate feature data is the feature exchanged by different participants during the inference phase of VFL. Experiments on four datasets demonstrate that our methods significantly outperform state-of-the-art techniques in attack precision. Our work exposes severe privacy vulnerabilities within VFL systems that pose real threats to practical VFL applications and thus confirms the necessity of further enhancing privacy protection in the VFL architecture.



## **5. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2401.07867v2) [paper-pdf](http://arxiv.org/pdf/2401.07867v2)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).



## **6. A Survey of Fragile Model Watermarking**

cs.CR

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.04809v2) [paper-pdf](http://arxiv.org/pdf/2406.04809v2)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.



## **7. Adversarial Attacks on Large Language Models in Medicine**

cs.AI

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12259v1) [paper-pdf](http://arxiv.org/pdf/2406.12259v1)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.



## **8. Privacy-Preserved Neural Graph Databases**

cs.DB

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2312.15591v5) [paper-pdf](http://arxiv.org/pdf/2312.15591v5)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Zihao Wang, Yangqiu Song

**Abstract**: In the era of large language models (LLMs), efficient and accurate data retrieval has become increasingly crucial for the use of domain-specific or private data in the retrieval augmented generation (RAG). Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (GDBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data which can be adaptively trained with LLMs. The usage of neural embedding storage and Complex neural logical Query Answering (CQA) provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the domain-specific or private databases. Malicious attackers can infer more sensitive information in the database using well-designed queries such as from the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training stage due to the privacy concerns. In this work, we propose a privacy-preserved neural graph database (P-NGDB) framework to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to enforce the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries.



## **9. Robust Text Classification: Analyzing Prototype-Based Networks**

cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2311.06647v2) [paper-pdf](http://arxiv.org/pdf/2311.06647v2)

**Authors**: Zhivar Sourati, Darshan Deshpande, Filip Ilievski, Kiril Gashteovski, Sascha Saralajew

**Abstract**: Downstream applications often require text classification models to be accurate and robust. While the accuracy of the state-of-the-art Language Models (LMs) approximates human performance, they often exhibit a drop in performance on noisy data found in the real world. This lack of robustness can be concerning, as even small perturbations in the text, irrelevant to the target task, can cause classifiers to incorrectly change their predictions. A potential solution can be the family of Prototype-Based Networks (PBNs) that classifies examples based on their similarity to prototypical examples of a class (prototypes) and has been shown to be robust to noise for computer vision tasks. In this paper, we study whether the robustness properties of PBNs transfer to text classification tasks under both targeted and static adversarial attack settings. Our results show that PBNs, as a mere architectural variation of vanilla LMs, offer more robustness compared to vanilla LMs under both targeted and static settings. We showcase how PBNs' interpretability can help us to understand PBNs' robustness properties. Finally, our ablation studies reveal the sensitivity of PBNs' robustness to how strictly clustering is done in the training phase, as tighter clustering results in less robust PBNs.



## **10. BadSampler: Harnessing the Power of Catastrophic Forgetting to Poison Byzantine-robust Federated Learning**

cs.CR

In Proceedings of the 30th ACM SIGKDD Conference on Knowledge  Discovery and Data Mining (KDD' 24), August 25-29, 2024, Barcelona, Spain

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12222v1) [paper-pdf](http://arxiv.org/pdf/2406.12222v1)

**Authors**: Yi Liu, Cong Wang, Xingliang Yuan

**Abstract**: Federated Learning (FL) is susceptible to poisoning attacks, wherein compromised clients manipulate the global model by modifying local datasets or sending manipulated model updates. Experienced defenders can readily detect and mitigate the poisoning effects of malicious behaviors using Byzantine-robust aggregation rules. However, the exploration of poisoning attacks in scenarios where such behaviors are absent remains largely unexplored for Byzantine-robust FL. This paper addresses the challenging problem of poisoning Byzantine-robust FL by introducing catastrophic forgetting. To fill this gap, we first formally define generalization error and establish its connection to catastrophic forgetting, paving the way for the development of a clean-label data poisoning attack named BadSampler. This attack leverages only clean-label data (i.e., without poisoned data) to poison Byzantine-robust FL and requires the adversary to selectively sample training data with high loss to feed model training and maximize the model's generalization error. We formulate the attack as an optimization problem and present two elegant adversarial sampling strategies, Top-$\kappa$ sampling, and meta-sampling, to approximately solve it. Additionally, our formal error upper bound and time complexity analysis demonstrate that our design can preserve attack utility with high efficiency. Extensive evaluations on two real-world datasets illustrate the effectiveness and performance of our proposed attacks.



## **11. Attack on Scene Flow using Point Clouds**

cs.CV

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2404.13621v3) [paper-pdf](http://arxiv.org/pdf/2404.13621v3)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. The robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants shows a higher vulnerability for the optical flow networks.



## **12. Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models**

cs.LG

ICML 2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2402.02207v2) [paper-pdf](http://arxiv.org/pdf/2402.02207v2)

**Authors**: Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, Timothy Hospedales

**Abstract**: Current vision large language models (VLLMs) exhibit remarkable capabilities yet are prone to generate harmful content and are vulnerable to even the simplest jailbreaking attacks. Our initial analysis finds that this is due to the presence of harmful data during vision-language instruction fine-tuning, and that VLLM fine-tuning can cause forgetting of safety alignment previously learned by the underpinning LLM. To address this issue, we first curate a vision-language safe instruction-following dataset VLGuard covering various harmful categories. Our experiments demonstrate that integrating this dataset into standard vision-language fine-tuning or utilizing it for post-hoc fine-tuning effectively safety aligns VLLMs. This alignment is achieved with minimal impact on, or even enhancement of, the models' helpfulness. The versatility of our safety fine-tuning dataset makes it a valuable resource for safety-testing existing VLLMs, training new models or safeguarding pre-trained VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject unsafe instructions and substantially reduce the success rates of several black-box adversarial attacks, which approach zero in many cases. The code and dataset are available at https://github.com/ys-zong/VLGuard.



## **13. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.08298v2) [paper-pdf](http://arxiv.org/pdf/2406.08298v2)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.



## **14. Threat analysis and adversarial model for Smart Grids**

cs.CR

Presented at the Workshop on Attackers and Cyber-Crime Operations  (WACCO). More details available at https://wacco-workshop.org

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11716v1) [paper-pdf](http://arxiv.org/pdf/2406.11716v1)

**Authors**: Javier Sande Ríos, Jesús Canal Sánchez, Carmen Manzano Hernandez, Sergio Pastrana

**Abstract**: The power grid is a critical infrastructure that allows for the efficient and robust generation, transmission, delivery and consumption of electricity. In the recent years, the physical components have been equipped with computing and network devices, which optimizes the operation and maintenance of the grid. The cyber domain of this smart power grid opens a new plethora of threats, which adds to classical threats on the physical domain. Accordingly, different stakeholders including regulation bodies, industry and academy, are making increasing efforts to provide security mechanisms to mitigate and reduce cyber-risks. Despite these efforts, there have been various cyberattacks that have affected the smart grid, leading in some cases to catastrophic consequences, showcasing that the industry might not be prepared for attacks from high profile adversaries. At the same time, recent work shows a lack of agreement among grid practitioners and academic experts on the feasibility and consequences of academic-proposed threats. This is in part due to inadequate simulation models which do not evaluate threats based on attackers full capabilities and goals. To address this gap, in this work we first analyze the main attack surfaces of the smart grid, and then conduct a threat analysis from the adversarial model perspective, including different levels of knowledge, goals, motivations and capabilities. To validate the model, we provide real-world examples of the potential capabilities by studying known vulnerabilities in critical components, and then analyzing existing cyber-attacks that have affected the smart grid, either directly or indirectly.



## **15. Harmonizing Feature Maps: A Graph Convolutional Approach for Enhancing Adversarial Robustness**

cs.CV

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11576v1) [paper-pdf](http://arxiv.org/pdf/2406.11576v1)

**Authors**: Kejia Zhang, Juanjuan Weng, Junwei Wu, Guoqing Yang, Shaozi Li, Zhiming Luo

**Abstract**: The vulnerability of Deep Neural Networks to adversarial perturbations presents significant security concerns, as the imperceptible perturbations can contaminate the feature space and lead to incorrect predictions. Recent studies have attempted to calibrate contaminated features by either suppressing or over-activating particular channels. Despite these efforts, we claim that adversarial attacks exhibit varying disruption levels across individual channels. Furthermore, we argue that harmonizing feature maps via graph and employing graph convolution can calibrate contaminated features. To this end, we introduce an innovative plug-and-play module called Feature Map-based Reconstructed Graph Convolution (FMR-GC). FMR-GC harmonizes feature maps in the channel dimension to reconstruct the graph, then employs graph convolution to capture neighborhood information, effectively calibrating contaminated features. Extensive experiments have demonstrated the superior performance and scalability of FMR-GC. Moreover, our model can be combined with advanced adversarial training methods to considerably enhance robustness without compromising the model's clean accuracy.



## **16. Do Parameters Reveal More than Loss for Membership Inference?**

cs.LG

Accepted at High-dimensional Learning Dynamics (HiLD) Workshop, ICML  2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11544v1) [paper-pdf](http://arxiv.org/pdf/2406.11544v1)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks aim to infer whether an individual record was used to train a model, serving as a key tool for disclosure auditing. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide very tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for most useful settings such as stochastic gradient descent, and that optimal membership inference indeed requires white-box access. We validate our findings with a new white-box inference attack IHA (Inverse Hessian Attack) that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both audits and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership privacy auditing.



## **17. FullCert: Deterministic End-to-End Certification for Training and Inference of Neural Networks**

cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11522v1) [paper-pdf](http://arxiv.org/pdf/2406.11522v1)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstract**: Modern machine learning models are sensitive to the manipulation of both the training data (poisoning attacks) and inference data (adversarial examples). Recognizing this issue, the community has developed many empirical defenses against both attacks and, more recently, provable certification methods against inference-time attacks. However, such guarantees are still largely lacking for training-time attacks. In this work, we present FullCert, the first end-to-end certifier with sound, deterministic bounds, which proves robustness against both training-time and inference-time attacks. We first bound all possible perturbations an adversary can make to the training data under the considered threat model. Using these constraints, we bound the perturbations' influence on the model's parameters. Finally, we bound the impact of these parameter changes on the model's prediction, resulting in joint robustness guarantees against poisoning and adversarial examples. To facilitate this novel certification paradigm, we combine our theoretical work with a new open-source library BoundFlow, which enables model training on bounded datasets. We experimentally demonstrate FullCert's feasibility on two different datasets.



## **18. Obfuscating IoT Device Scanning Activity via Adversarial Example Generation**

cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11515v1) [paper-pdf](http://arxiv.org/pdf/2406.11515v1)

**Authors**: Haocong Li, Yaxin Zhang, Long Cheng, Wenjia Niu, Haining Wang, Qiang Li

**Abstract**: Nowadays, attackers target Internet of Things (IoT) devices for security exploitation, and search engines for devices and services compromise user privacy, including IP addresses, open ports, device types, vendors, and products.Typically, application banners are used to recognize IoT device profiles during network measurement and reconnaissance. In this paper, we propose a novel approach to obfuscating IoT device banners (BANADV) based on adversarial examples. The key idea is to explore the susceptibility of fingerprinting techniques to a slight perturbation of an IoT device banner. By modifying device banners, BANADV disrupts the collection of IoT device profiles. To validate the efficacy of BANADV, we conduct a set of experiments. Our evaluation results show that adversarial examples can spoof state-of-the-art fingerprinting techniques, including learning- and matching-based approaches. We further provide a detailed analysis of the weakness of learning-based/matching-based fingerprints to carefully crafted samples. Overall, the innovations of BANADV lie in three aspects: (1) it utilizes an IoT-related semantic space and a visual similarity space to locate available manipulating perturbations of IoT banners; (2) it achieves at least 80\% success rate for spoofing IoT scanning techniques; and (3) it is the first to utilize adversarial examples of IoT banners in network measurement and reconnaissance.



## **19. Adapters Mixup: Mixing Parameter-Efficient Adapters to Enhance the Adversarial Robustness of Fine-tuned Pre-trained Text Classifiers**

cs.CL

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2401.10111v2) [paper-pdf](http://arxiv.org/pdf/2401.10111v2)

**Authors**: Tuc Nguyen, Thai Le

**Abstract**: Existing works show that augmenting the training data of pre-trained language models (PLMs) for classification tasks fine-tuned via parameter-efficient fine-tuning methods (PEFT) using both clean and adversarial examples can enhance their robustness under adversarial attacks. However, this adversarial training paradigm often leads to performance degradation on clean inputs and requires frequent re-training on the entire data to account for new, unknown attacks. To overcome these challenges while still harnessing the benefits of adversarial training and the efficiency of PEFT, this work proposes a novel approach, called AdpMixup, that combines two paradigms: (1) fine-tuning through adapters and (2) adversarial augmentation via mixup to dynamically leverage existing knowledge from a set of pre-known attacks for robust inference. Intuitively, AdpMixup fine-tunes PLMs with multiple adapters with both clean and pre-known adversarial examples and intelligently mixes them up in different ratios during prediction. Our experiments show AdpMixup achieves the best trade-off between training efficiency and robustness under both pre-known and unknown attacks, compared to existing baselines on five downstream tasks across six varied black-box attacks and 2 PLMs. All source code will be available.



## **20. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2405.19360v2) [paper-pdf](http://arxiv.org/pdf/2405.19360v2)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.



## **21. $\texttt{MoE-RBench}$: Towards Building Reliable Language Models with Sparse Mixture-of-Experts**

cs.LG

9 pages, 8 figures, camera ready on ICML2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11353v1) [paper-pdf](http://arxiv.org/pdf/2406.11353v1)

**Authors**: Guanjie Chen, Xinyu Zhao, Tianlong Chen, Yu Cheng

**Abstract**: Mixture-of-Experts (MoE) has gained increasing popularity as a promising framework for scaling up large language models (LLMs). However, the reliability assessment of MoE lags behind its surging applications. Moreover, when transferred to new domains such as in fine-tuning MoE models sometimes underperform their dense counterparts. Motivated by the research gap and counter-intuitive phenomenon, we propose $\texttt{MoE-RBench}$, the first comprehensive assessment of SMoE reliability from three aspects: $\textit{(i)}$ safety and hallucination, $\textit{(ii)}$ resilience to adversarial attacks, and $\textit{(iii)}$ out-of-distribution robustness. Extensive models and datasets are tested to compare the MoE to dense networks from these reliability dimensions. Our empirical observations suggest that with appropriate hyperparameters, training recipes, and inference techniques, we can build the MoE model more reliably than the dense LLM. In particular, we find that the robustness of SMoE is sensitive to the basic training settings. We hope that this study can provide deeper insights into how to adapt the pre-trained MoE model to other tasks with higher-generation security, quality, and stability. Codes are available at https://github.com/UNITES-Lab/MoE-RBench



## **22. Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game**

cs.GT

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2305.12872v3) [paper-pdf](http://arxiv.org/pdf/2305.12872v3)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Ruixiao Xu, Xin Yu, Jiakai Wang, Aishan Liu, Yaodong Yang, Xianglong Liu

**Abstract**: In this study, we explore the robustness of cooperative multi-agent reinforcement learning (c-MARL) against Byzantine failures, where any agent can enact arbitrary, worst-case actions due to malfunction or adversarial attack. To address the uncertainty that any agent can be adversarial, we propose a Bayesian Adversarial Robust Dec-POMDP (BARDec-POMDP) framework, which views Byzantine adversaries as nature-dictated types, represented by a separate transition. This allows agents to learn policies grounded on their posterior beliefs about the type of other agents, fostering collaboration with identified allies and minimizing vulnerability to adversarial manipulation. We define the optimal solution to the BARDec-POMDP as an ex post robust Bayesian Markov perfect equilibrium, which we proof to exist and weakly dominates the equilibrium of previous robust MARL approaches. To realize this equilibrium, we put forward a two-timescale actor-critic algorithm with almost sure convergence under specific conditions. Experimentation on matrix games, level-based foraging and StarCraft II indicate that, even under worst-case perturbations, our method successfully acquires intricate micromanagement skills and adaptively aligns with allies, demonstrating resilience against non-oblivious adversaries, random allies, observation-based attacks, and transfer-based attacks.



## **23. Optimal Attack and Defense for Reinforcement Learning**

cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2312.00198v2) [paper-pdf](http://arxiv.org/pdf/2312.00198v2)

**Authors**: Jeremy McMahan, Young Wu, Xiaojin Zhu, Qiaomin Xie

**Abstract**: To ensure the usefulness of Reinforcement Learning (RL) in real systems, it is crucial to ensure they are robust to noise and adversarial attacks. In adversarial RL, an external attacker has the power to manipulate the victim agent's interaction with the environment. We study the full class of online manipulation attacks, which include (i) state attacks, (ii) observation attacks (which are a generalization of perceived-state attacks), (iii) action attacks, and (iv) reward attacks. We show the attacker's problem of designing a stealthy attack that maximizes its own expected reward, which often corresponds to minimizing the victim's value, is captured by a Markov Decision Process (MDP) that we call a meta-MDP since it is not the true environment but a higher level environment induced by the attacked interaction. We show that the attacker can derive optimal attacks by planning in polynomial time or learning with polynomial sample complexity using standard RL techniques. We argue that the optimal defense policy for the victim can be computed as the solution to a stochastic Stackelberg game, which can be further simplified into a partially-observable turn-based stochastic game (POTBSG). Neither the attacker nor the victim would benefit from deviating from their respective optimal policies, thus such solutions are truly robust. Although the defense problem is NP-hard, we show that optimal Markovian defenses can be computed (learned) in polynomial time (sample complexity) in many scenarios.



## **24. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

cs.CL

8 pages

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11260v1) [paper-pdf](http://arxiv.org/pdf/2406.11260v1)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.



## **25. The Benefits of Power Regularization in Cooperative Reinforcement Learning**

cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11240v1) [paper-pdf](http://arxiv.org/pdf/2406.11240v1)

**Authors**: Michelle Li, Michael Dennis

**Abstract**: Cooperative Multi-Agent Reinforcement Learning (MARL) algorithms, trained only to optimize task reward, can lead to a concentration of power where the failure or adversarial intent of a single agent could decimate the reward of every agent in the system. In the context of teams of people, it is often useful to explicitly consider how power is distributed to ensure no person becomes a single point of failure. Here, we argue that explicitly regularizing the concentration of power in cooperative RL systems can result in systems which are more robust to single agent failure, adversarial attacks, and incentive changes of co-players. To this end, we define a practical pairwise measure of power that captures the ability of any co-player to influence the ego agent's reward, and then propose a power-regularized objective which balances task reward and power concentration. Given this new objective, we show that there always exists an equilibrium where every agent is playing a power-regularized best-response balancing power and task reward. Moreover, we present two algorithms for training agents towards this power-regularized objective: Sample Based Power Regularization (SBPR), which injects adversarial data during training; and Power Regularization via Intrinsic Motivation (PRIM), which adds an intrinsic motivation to regulate power to the training objective. Our experiments demonstrate that both algorithms successfully balance task reward and power, leading to lower power behavior than the baseline of task-only reward and avoid catastrophic events in case an agent in the system goes off-policy.



## **26. Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics**

cs.RO

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2402.10340v4) [paper-pdf](http://arxiv.org/pdf/2402.10340v4)

**Authors**: Xiyang Wu, Souradip Chakraborty, Ruiqi Xian, Jing Liang, Tianrui Guan, Fuxiao Liu, Brian M. Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works focus on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation and navigation. Despite these improvements, analyzing the safety of such systems remains underexplored yet extremely critical. LLMs and VLMs are highly susceptible to adversarial inputs, prompting a significant inquiry into the safety of robotic systems. This concern is important because robotics operate in the physical world where erroneous actions can result in severe consequences. This paper explores this issue thoroughly, presenting a mathematical formulation of potential attacks on LLM/VLM-based robotic systems and offering experimental evidence of the safety challenges. Our empirical findings highlight a significant vulnerability: simple modifications to the input can drastically reduce system effectiveness. Specifically, our results demonstrate an average performance deterioration of 19.4% under minor input prompt modifications and a more alarming 29.1% under slight perceptual changes. These findings underscore the urgent need for robust countermeasures to ensure the safe and reliable deployment of advanced LLM/VLM-based robotic systems.



## **27. garak: A Framework for Security Probing Large Language Models**

cs.CL

https://garak.ai

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.11036v1) [paper-pdf](http://arxiv.org/pdf/2406.11036v1)

**Authors**: Leon Derczynski, Erick Galinkin, Jeffrey Martin, Subho Majumdar, Nanna Inie

**Abstract**: As Large Language Models (LLMs) are deployed and integrated into thousands of applications, the need for scalable evaluation of how models respond to adversarial attacks grows rapidly. However, LLM security is a moving target: models produce unpredictable output, are constantly updated, and the potential adversary is highly diverse: anyone with access to the internet and a decent command of natural language. Further, what constitutes a security weak in one context may not be an issue in a different context; one-fits-all guardrails remain theoretical. In this paper, we argue that it is time to rethink what constitutes ``LLM security'', and pursue a holistic approach to LLM security evaluation, where exploration and discovery of issues are central. To this end, this paper introduces garak (Generative AI Red-teaming and Assessment Kit), a framework which can be used to discover and identify vulnerabilities in a target LLM or dialog system. garak probes an LLM in a structured fashion to discover potential vulnerabilities. The outputs of the framework describe a target model's weaknesses, contribute to an informed discussion of what composes vulnerabilities in unique contexts, and can inform alignment and policy discussions for LLM deployment.



## **28. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2404.01318v3) [paper-pdf](http://arxiv.org/pdf/2404.01318v3)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.



## **29. Adversarial Illusions in Multi-Modal Embeddings**

cs.CR

In USENIX Security'24

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2308.11804v4) [paper-pdf](http://arxiv.org/pdf/2308.11804v4)

**Authors**: Tingwei Zhang, Rishi Jha, Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal embeddings encode texts, images, thermal images, sounds, and videos into a single embedding space, aligning representations across different modalities (e.g., associate an image of a dog with a barking sound). In this paper, we show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an image or a sound, an adversary can perturb it to make its embedding close to an arbitrary, adversary-chosen input in another modality.   These attacks are cross-modal and targeted: the adversary can align any image or sound with any target of his choice. Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks and modalities, enabling a wholesale compromise of current and future tasks, as well as modalities not available to the adversary. Using ImageBind and AudioCLIP embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, zero-shot classification, and audio retrieval.   We investigate transferability of illusions across different embeddings and develop a black-box version of our method that we use to demonstrate the first adversarial alignment attack on Amazon's commercial, proprietary Titan embedding. Finally, we analyze countermeasures and evasion attacks.



## **30. ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator**

cs.CL

18 pages, 7 figures

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2405.18111v2) [paper-pdf](http://arxiv.org/pdf/2405.18111v2)

**Authors**: Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, Lei Sha

**Abstract**: Large language models (LLMs) are proven to benefit a lot from retrieval-augmented generation (RAG) in alleviating hallucinations confronted with knowledge-intensive questions. RAG adopts information retrieval techniques to inject external knowledge from semantic-relevant documents as input contexts. However, due to today's Internet being flooded with numerous noisy and fabricating content, it is inevitable that RAG systems are vulnerable to these noises and prone to respond incorrectly. To this end, we propose to optimize the retrieval-augmented Generator with a Adversarial Tuning Multi-agent system (ATM). The ATM steers the Generator to have a robust perspective of useful documents for question answering with the help of an auxiliary Attacker agent. The Generator and the Attacker are tuned adversarially for several iterations. After rounds of multi-agent iterative tuning, the Generator can eventually better discriminate useful documents amongst fabrications. The experimental results verify the effectiveness of ATM and we also observe that the Generator can achieve better performance compared to state-of-the-art baselines.



## **31. RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models**

cs.CL

48 pages, 7 figures, 12 tables

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10890v1) [paper-pdf](http://arxiv.org/pdf/2406.10890v1)

**Authors**: Zhuoran Jin, Pengfei Cao, Chenhao Wang, Zhitao He, Hongbang Yuan, Jiachun Li, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: Large language models (LLMs) inevitably memorize sensitive, copyrighted, and harmful knowledge from the training corpus; therefore, it is crucial to erase this knowledge from the models. Machine unlearning is a promising solution for efficiently removing specific knowledge by post hoc modifying models. In this paper, we propose a Real-World Knowledge Unlearning benchmark (RWKU) for LLM unlearning. RWKU is designed based on the following three key factors: (1) For the task setting, we consider a more practical and challenging unlearning setting, where neither the forget corpus nor the retain corpus is accessible. (2) For the knowledge source, we choose 200 real-world famous people as the unlearning targets and show that such popular knowledge is widely present in various LLMs. (3) For the evaluation framework, we design the forget set and the retain set to evaluate the model's capabilities across various real-world applications. Regarding the forget set, we provide four four membership inference attack (MIA) methods and nine kinds of adversarial attack probes to rigorously test unlearning efficacy. Regarding the retain set, we assess locality and utility in terms of neighbor perturbation, general ability, reasoning ability, truthfulness, factuality, and fluency. We conduct extensive experiments across two unlearning scenarios, two models and six baseline methods and obtain some meaningful findings. We release our benchmark and code publicly at http://rwku-bench.github.io for future work.



## **32. Imperceptible Face Forgery Attack via Adversarial Semantic Mask**

cs.CV

The code is publicly available

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10887v1) [paper-pdf](http://arxiv.org/pdf/2406.10887v1)

**Authors**: Decheng Liu, Qixuan Su, Chunlei Peng, Nannan Wang, Xinbo Gao

**Abstract**: With the great development of generative model techniques, face forgery detection draws more and more attention in the related field. Researchers find that existing face forgery models are still vulnerable to adversarial examples with generated pixel perturbations in the global image. These generated adversarial samples still can't achieve satisfactory performance because of the high detectability. To address these problems, we propose an Adversarial Semantic Mask Attack framework (ASMA) which can generate adversarial examples with good transferability and invisibility. Specifically, we propose a novel adversarial semantic mask generative model, which can constrain generated perturbations in local semantic regions for good stealthiness. The designed adaptive semantic mask selection strategy can effectively leverage the class activation values of different semantic regions, and further ensure better attack transferability and stealthiness. Extensive experiments on the public face forgery dataset prove the proposed method achieves superior performance compared with several representative adversarial attack methods. The code is publicly available at https://github.com/clawerO-O/ASMA.



## **33. SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems**

cs.LG

To appear in the ACM Conference on Computer and Communications  Security (CCS'24), October 14-18, 2024, Salt Lake City, UT, USA

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2402.03741v2) [paper-pdf](http://arxiv.org/pdf/2402.03741v2)

**Authors**: Oubo Ma, Yuwen Pu, Linkang Du, Yang Dai, Ruo Wang, Xiaolei Liu, Yingcai Wu, Shouling Ji

**Abstract**: Recent advancements in multi-agent reinforcement learning (MARL) have opened up vast application prospects, such as swarm control of drones, collaborative manipulation by robotic arms, and multi-target encirclement. However, potential security threats during the MARL deployment need more attention and thorough investigation. Recent research reveals that attackers can rapidly exploit the victim's vulnerabilities, generating adversarial policies that result in the failure of specific tasks. For instance, reducing the winning rate of a superhuman-level Go AI to around 20%. Existing studies predominantly focus on two-player competitive environments, assuming attackers possess complete global state observation.   In this study, we unveil, for the first time, the capability of attackers to generate adversarial policies even when restricted to partial observations of the victims in multi-agent competitive environments. Specifically, we propose a novel black-box attack (SUB-PLAY) that incorporates the concept of constructing multiple subgames to mitigate the impact of partial observability and suggests sharing transitions among subpolicies to improve attackers' exploitative ability. Extensive evaluations demonstrate the effectiveness of SUB-PLAY under three typical partial observability limitations. Visualization results indicate that adversarial policies induce significantly different activations of the victims' policy networks. Furthermore, we evaluate three potential defenses aimed at exploring ways to mitigate security threats posed by adversarial policies, providing constructive recommendations for deploying MARL in competitive environments.



## **34. Mitigating Accuracy-Robustness Trade-off via Balanced Multi-Teacher Adversarial Distillation**

cs.LG

Accepted by TPAMI2024

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2306.16170v3) [paper-pdf](http://arxiv.org/pdf/2306.16170v3)

**Authors**: Shiji Zhao, Xizhe Wang, Xingxing Wei

**Abstract**: Adversarial Training is a practical approach for improving the robustness of deep neural networks against adversarial attacks. Although bringing reliable robustness, the performance towards clean examples is negatively affected after Adversarial Training, which means a trade-off exists between accuracy and robustness. Recently, some studies have tried to use knowledge distillation methods in Adversarial Training, achieving competitive performance in improving the robustness but the accuracy for clean samples is still limited. In this paper, to mitigate the accuracy-robustness trade-off, we introduce the Balanced Multi-Teacher Adversarial Robustness Distillation (B-MTARD) to guide the model's Adversarial Training process by applying a strong clean teacher and a strong robust teacher to handle the clean examples and adversarial examples, respectively. During the optimization process, to ensure that different teachers show similar knowledge scales, we design the Entropy-Based Balance algorithm to adjust the teacher's temperature and keep the teachers' information entropy consistent. Besides, to ensure that the student has a relatively consistent learning speed from multiple teachers, we propose the Normalization Loss Balance algorithm to adjust the learning weights of different types of knowledge. A series of experiments conducted on three public datasets demonstrate that B-MTARD outperforms the state-of-the-art methods against various adversarial attacks.



## **35. KGPA: Robustness Evaluation for Large Language Models via Cross-Domain Knowledge Graphs**

cs.CL

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10802v1) [paper-pdf](http://arxiv.org/pdf/2406.10802v1)

**Authors**: Aihua Pei, Zehua Yang, Shunan Zhu, Ruoxi Cheng, Ju Jia, Lina Wang

**Abstract**: Existing frameworks for assessing robustness of large language models (LLMs) overly depend on specific benchmarks, increasing costs and failing to evaluate performance of LLMs in professional domains due to dataset limitations. This paper proposes a framework that systematically evaluates the robustness of LLMs under adversarial attack scenarios by leveraging knowledge graphs (KGs). Our framework generates original prompts from the triplets of knowledge graphs and creates adversarial prompts by poisoning, assessing the robustness of LLMs through the results of these adversarial attacks. We systematically evaluate the effectiveness of this framework and its modules. Experiments show that adversarial robustness of the ChatGPT family ranks as GPT-4-turbo > GPT-4o > GPT-3.5-turbo, and the robustness of large language models is influenced by the professional domains in which they operate.



## **36. Adversarial Math Word Problem Generation**

cs.CL

Code/data: https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2402.17916v3) [paper-pdf](http://arxiv.org/pdf/2402.17916v3)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis to investigate the cause of failure, providing further insights into the limitations of LLMs.



## **37. Federated Multi-Armed Bandits Under Byzantine Attacks**

cs.LG

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2205.04134v2) [paper-pdf](http://arxiv.org/pdf/2205.04134v2)

**Authors**: Artun Saday, İlker Demirel, Yiğit Yıldırım, Cem Tekin

**Abstract**: Multi-armed bandits (MAB) is a sequential decision-making model in which the learner controls the trade-off between exploration and exploitation to maximize its cumulative reward. Federated multi-armed bandits (FMAB) is an emerging framework where a cohort of learners with heterogeneous local models play an MAB game and communicate their aggregated feedback to a server to learn a globally optimal arm. Two key hurdles in FMAB are communication-efficient learning and resilience to adversarial attacks. To address these issues, we study the FMAB problem in the presence of Byzantine clients who can send false model updates threatening the learning process. We analyze the sample complexity and the regret of $\beta$-optimal arm identification. We borrow tools from robust statistics and propose a median-of-means (MoM)-based online algorithm, Fed-MoM-UCB, to cope with Byzantine clients. In particular, we show that if the Byzantine clients constitute less than half of the cohort, the cumulative regret with respect to $\beta$-optimal arms is bounded over time with high probability, showcasing both communication efficiency and Byzantine resilience. We analyze the interplay between the algorithm parameters, a discernibility margin, regret, communication cost, and the arms' suboptimality gaps. We demonstrate Fed-MoM-UCB's effectiveness against the baselines in the presence of Byzantine attacks via experiments.



## **38. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

cs.CR

Stochastic investment models and a Bayesian approach to better  modeling of uncertainty : adversarial machine learning or Stochastic market

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2406.10719v1) [paper-pdf](http://arxiv.org/pdf/2406.10719v1)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.



## **39. Hijacking Large Language Models via Adversarial In-Context Learning**

cs.LG

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2311.09948v2) [paper-pdf](http://arxiv.org/pdf/2311.09948v2)

**Authors**: Yao Qiang, Xiangyu Zhou, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the precondition prompts. Despite its promising performance, ICL suffers from instability with the choice and arrangement of examples. Additionally, crafted adversarial attacks pose a notable threat to the robustness of ICL. However, existing attacks are either easy to detect, rely on external models, or lack specificity towards ICL. This work introduces a novel transferable attack against ICL to address these issues, aiming to hijack LLMs to generate the target response or jailbreak. Our hijacking attack leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos without directly contaminating the user queries. Comprehensive experimental results across different generation and jailbreaking tasks highlight the effectiveness of our hijacking attack, resulting in distracted attention towards adversarial tokens and consequently leading to unwanted target outputs. We also propose a defense strategy against hijacking attacks through the use of extra clean demos, which enhances the robustness of LLMs during ICL. Broadly, this work reveals the significant security vulnerabilities of LLMs and emphasizes the necessity for in-depth studies on their robustness.



## **40. E-SAGE: Explainability-based Defense Against Backdoor Attacks on Graph Neural Networks**

cs.CR

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2406.10655v1) [paper-pdf](http://arxiv.org/pdf/2406.10655v1)

**Authors**: Dingqiang Yuan, Xiaohua Xu, Lei Yu, Tongchang Han, Rongchang Li, Meng Han

**Abstract**: Graph Neural Networks (GNNs) have recently been widely adopted in multiple domains. Yet, they are notably vulnerable to adversarial and backdoor attacks. In particular, backdoor attacks based on subgraph insertion have been shown to be effective in graph classification tasks while being stealthy, successfully circumventing various existing defense methods. In this paper, we propose E-SAGE, a novel approach to defending GNN backdoor attacks based on explainability. We find that the malicious edges and benign edges have significant differences in the importance scores for explainability evaluation. Accordingly, E-SAGE adaptively applies an iterative edge pruning process on the graph based on the edge scores. Through extensive experiments, we demonstrate the effectiveness of E-SAGE against state-of-the-art graph backdoor attacks in different attack settings. In addition, we investigate the effectiveness of E-SAGE against adversarial attacks.



## **41. From Trojan Horses to Castle Walls: Unveiling Bilateral Data Poisoning Effects in Diffusion Models**

cs.LG

9 pages, 5 figures, 4 tables

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2311.02373v2) [paper-pdf](http://arxiv.org/pdf/2311.02373v2)

**Authors**: Zhuoshi Pan, Yuguang Yao, Gaowen Liu, Bingquan Shen, H. Vicky Zhao, Ramana Rao Kompella, Sijia Liu

**Abstract**: While state-of-the-art diffusion models (DMs) excel in image generation, concerns regarding their security persist. Earlier research highlighted DMs' vulnerability to data poisoning attacks, but these studies placed stricter requirements than conventional methods like `BadNets' in image classification. This is because the art necessitates modifications to the diffusion training and sampling procedures. Unlike the prior work, we investigate whether BadNets-like data poisoning methods can directly degrade the generation by DMs. In other words, if only the training dataset is contaminated (without manipulating the diffusion process), how will this affect the performance of learned DMs? In this setting, we uncover bilateral data poisoning effects that not only serve an adversarial purpose (compromising the functionality of DMs) but also offer a defensive advantage (which can be leveraged for defense in classification tasks against poisoning attacks). We show that a BadNets-like data poisoning attack remains effective in DMs for producing incorrect images (misaligned with the intended text conditions). Meanwhile, poisoned DMs exhibit an increased ratio of triggers, a phenomenon we refer to as `trigger amplification', among the generated images. This insight can be then used to enhance the detection of poisoned training data. In addition, even under a low poisoning ratio, studying the poisoning effects of DMs is also valuable for designing robust image classifiers against such attacks. Last but not least, we establish a meaningful linkage between data poisoning and the phenomenon of data replications by exploring DMs' inherent data memorization tendencies.



## **42. Robust Image Classification in the Presence of Out-of-Distribution and Adversarial Samples Using Attractors in Neural Networks**

cs.CV

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2406.10579v1) [paper-pdf](http://arxiv.org/pdf/2406.10579v1)

**Authors**: Nasrin Alipour, Seyyed Ali SeyyedSalehi

**Abstract**: The proper handling of out-of-distribution (OOD) samples in deep classifiers is a critical concern for ensuring the suitability of deep neural networks in safety-critical systems. Existing approaches developed for robust OOD detection in the presence of adversarial attacks lose their performance by increasing the perturbation levels. This study proposes a method for robust classification in the presence of OOD samples and adversarial attacks with high perturbation levels. The proposed approach utilizes a fully connected neural network that is trained to use training samples as its attractors, enhancing its robustness. This network has the ability to classify inputs and identify OOD samples as well. To evaluate this method, the network is trained on the MNIST dataset, and its performance is tested on adversarial examples. The results indicate that the network maintains its performance even when classifying adversarial examples, achieving 87.13% accuracy when dealing with highly perturbed MNIST test data. Furthermore, by using fashion-MNIST and CIFAR-10-bw as OOD samples, the network can distinguish these samples from MNIST samples with an accuracy of 98.84% and 99.28%, respectively. In the presence of severe adversarial attacks, these measures decrease slightly to 98.48% and 98.88%, indicating the robustness of the proposed method.



## **43. Graph Neural Backdoor: Fundamentals, Methodologies, Applications, and Future Directions**

cs.LG

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2406.10573v1) [paper-pdf](http://arxiv.org/pdf/2406.10573v1)

**Authors**: Xiao Yang, Gaolei Li, Jianhua Li

**Abstract**: Graph Neural Networks (GNNs) have significantly advanced various downstream graph-relevant tasks, encompassing recommender systems, molecular structure prediction, social media analysis, etc. Despite the boosts of GNN, recent research has empirically demonstrated its potential vulnerability to backdoor attacks, wherein adversaries employ triggers to poison input samples, inducing GNN to adversary-premeditated malicious outputs. This is typically due to the controlled training process, or the deployment of untrusted models, such as delegating model training to third-party service, leveraging external training sets, and employing pre-trained models from online sources. Although there's an ongoing increase in research on GNN backdoors, comprehensive investigation into this field is lacking. To bridge this gap, we propose the first survey dedicated to GNN backdoors. We begin by outlining the fundamental definition of GNN, followed by the detailed summarization and categorization of current GNN backdoor attacks and defenses based on their technical characteristics and application scenarios. Subsequently, the analysis of the applicability and use cases of GNN backdoors is undertaken. Finally, the exploration of potential research directions of GNN backdoors is presented. This survey aims to explore the principles of graph backdoors, provide insights to defenders, and promote future security research.



## **44. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

cs.CV

Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-06-15    [abs](http://arxiv.org/abs/2310.11868v3) [paper-pdf](http://arxiv.org/pdf/2310.11868v3)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models.Through extensive benchmarking, we evaluate the robustness of five widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safety-driven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: This paper contains model outputs that may be offensive in nature.



## **45. Towards the Theory of Unsupervised Federated Learning: Non-asymptotic Analysis of Federated EM Algorithms**

stat.ML

50 pages, 3 figures

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2310.15330v3) [paper-pdf](http://arxiv.org/pdf/2310.15330v3)

**Authors**: Ye Tian, Haolei Weng, Yang Feng

**Abstract**: While supervised federated learning approaches have enjoyed significant success, the domain of unsupervised federated learning remains relatively underexplored. Several federated EM algorithms have gained popularity in practice, however, their theoretical foundations are often lacking. In this paper, we first introduce a federated gradient EM algorithm (FedGrEM) designed for the unsupervised learning of mixture models, which supplements the existing federated EM algorithms by considering task heterogeneity and potential adversarial attacks. We present a comprehensive finite-sample theory that holds for general mixture models, then apply this general theory on specific statistical models to characterize the explicit estimation error of model parameters and mixture proportions. Our theory elucidates when and how FedGrEM outperforms local single-task learning with insights extending to existing federated EM algorithms. This bridges the gap between their practical success and theoretical understanding. Our numerical results validate our theory, and demonstrate FedGrEM's superiority over existing unsupervised federated learning benchmarks.



## **46. Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models**

cs.CV

Codes are available at https://github.com/OPTML-Group/AdvUnlearn

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2405.15234v2) [paper-pdf](http://arxiv.org/pdf/2405.15234v2)

**Authors**: Yimeng Zhang, Xin Chen, Jinghan Jia, Yihua Zhang, Chongyu Fan, Jiancheng Liu, Mingyi Hong, Ke Ding, Sijia Liu

**Abstract**: Diffusion models (DMs) have achieved remarkable success in text-to-image generation, but they also pose safety risks, such as the potential generation of harmful content and copyright violations. The techniques of machine unlearning, also known as concept erasing, have been developed to address these risks. However, these techniques remain vulnerable to adversarial prompt attacks, which can prompt DMs post-unlearning to regenerate undesired images containing concepts (such as nudity) meant to be erased. This work aims to enhance the robustness of concept erasing by integrating the principle of adversarial training (AT) into machine unlearning, resulting in the robust unlearning framework referred to as AdvUnlearn. However, achieving this effectively and efficiently is highly nontrivial. First, we find that a straightforward implementation of AT compromises DMs' image generation quality post-unlearning. To address this, we develop a utility-retaining regularization on an additional retain set, optimizing the trade-off between concept erasure robustness and model utility in AdvUnlearn. Moreover, we identify the text encoder as a more suitable module for robustification compared to UNet, ensuring unlearning effectiveness. And the acquired text encoder can serve as a plug-and-play robust unlearner for various DM types. Empirically, we perform extensive experiments to demonstrate the robustness advantage of AdvUnlearn across various DM unlearning scenarios, including the erasure of nudity, objects, and style concepts. In addition to robustness, AdvUnlearn also achieves a balanced tradeoff with model utility. To our knowledge, this is the first work to systematically explore robust DM unlearning through AT, setting it apart from existing methods that overlook robustness in concept erasing. Codes are available at: https://github.com/OPTML-Group/AdvUnlearn



## **47. I Still See You: Why Existing IoT Traffic Reshaping Fails**

cs.CR

EWSN'24 paper accepted, to appear

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.10358v1) [paper-pdf](http://arxiv.org/pdf/2406.10358v1)

**Authors**: Su Wang, Keyang Yu, Qi Li, Dong Chen

**Abstract**: The Internet traffic data produced by the Internet of Things (IoT) devices are collected by Internet Service Providers (ISPs) and device manufacturers, and often shared with their third parties to maintain and enhance user services. Unfortunately, on-path adversaries could infer and fingerprint users' sensitive privacy information such as occupancy and user activities by analyzing these network traffic traces. While there's a growing body of literature on defending against this side-channel attack-malicious IoT traffic analytics (TA), there's currently no systematic method to compare and evaluate the comprehensiveness of these existing studies. To address this problem, we design a new low-cost, open-source system framework-IoT Traffic Exposure Monitoring Toolkit (ITEMTK) that enables people to comprehensively examine and validate prior attack models and their defending approaches. In particular, we also design a novel image-based attack capable of inferring sensitive user information, even when users employ the most robust preventative measures in their smart homes. Researchers could leverage our new image-based attack to systematize and understand the existing literature on IoT traffic analysis attacks and preventing studies. Our results show that current defending approaches are not sufficient to protect IoT device user privacy. IoT devices are significantly vulnerable to our new image-based user privacy inference attacks, posing a grave threat to IoT device user privacy. We also highlight potential future improvements to enhance the defending approaches. ITEMTK's flexibility allows other researchers for easy expansion by integrating new TA attack models and prevention methods to benchmark their future work.



## **48. Automated Design of Linear Bounding Functions for Sigmoidal Nonlinearities in Neural Networks**

cs.LG

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.10154v1) [paper-pdf](http://arxiv.org/pdf/2406.10154v1)

**Authors**: Matthias König, Xiyue Zhang, Holger H. Hoos, Marta Kwiatkowska, Jan N. van Rijn

**Abstract**: The ubiquity of deep learning algorithms in various applications has amplified the need for assuring their robustness against small input perturbations such as those occurring in adversarial attacks. Existing complete verification techniques offer provable guarantees for all robustness queries but struggle to scale beyond small neural networks. To overcome this computational intractability, incomplete verification methods often rely on convex relaxation to over-approximate the nonlinearities in neural networks. Progress in tighter approximations has been achieved for piecewise linear functions. However, robustness verification of neural networks for general activation functions (e.g., Sigmoid, Tanh) remains under-explored and poses new challenges. Typically, these networks are verified using convex relaxation techniques, which involve computing linear upper and lower bounds of the nonlinear activation functions. In this work, we propose a novel parameter search method to improve the quality of these linear approximations. Specifically, we show that using a simple search method, carefully adapted to the given verification problem through state-of-the-art algorithm configuration techniques, improves the average global lower bound by 25% on average over the current state of the art on several commonly used local robustness verification benchmarks.



## **49. Over-parameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis**

cs.LG

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2406.10090v1) [paper-pdf](http://arxiv.org/pdf/2406.10090v1)

**Authors**: Zhang Chen, Luca Demetrio, Srishti Gupta, Xiaoyi Feng, Zhaoqiang Xia, Antonio Emanuele Cinà, Maura Pintor, Luca Oneto, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Thanks to their extensive capacity, over-parameterized neural networks exhibit superior predictive capabilities and generalization. However, having a large parameter space is considered one of the main suspects of the neural networks' vulnerability to adversarial example -- input samples crafted ad-hoc to induce a desired misclassification. Relevant literature has claimed contradictory remarks in support of and against the robustness of over-parameterized networks. These contradictory findings might be due to the failure of the attack employed to evaluate the networks' robustness. Previous research has demonstrated that depending on the considered model, the algorithm employed to generate adversarial examples may not function properly, leading to overestimating the model's robustness. In this work, we empirically study the robustness of over-parameterized networks against adversarial examples. However, unlike the previous works, we also evaluate the considered attack's reliability to support the results' veracity. Our results show that over-parameterized networks are robust against adversarial attacks as opposed to their under-parameterized counterparts.



## **50. Perturbing Attention Gives You More Bang for the Buck: Subtle Imaging Perturbations That Efficiently Fool Customized Diffusion Models**

cs.CV

Published at CVPR 2024, code:https://github.com/CO2-cityao/CAAT

**SubmitDate**: 2024-06-14    [abs](http://arxiv.org/abs/2404.15081v2) [paper-pdf](http://arxiv.org/pdf/2404.15081v2)

**Authors**: Jingyao Xu, Yuetong Lu, Yandong Li, Siyang Lu, Dongdong Wang, Xiang Wei

**Abstract**: Diffusion models (DMs) embark a new era of generative modeling and offer more opportunities for efficient generating high-quality and realistic data samples. However, their widespread use has also brought forth new challenges in model security, which motivates the creation of more effective adversarial attackers on DMs to understand its vulnerability. We propose CAAT, a simple but generic and efficient approach that does not require costly training to effectively fool latent diffusion models (LDMs). The approach is based on the observation that cross-attention layers exhibits higher sensitivity to gradient change, allowing for leveraging subtle perturbations on published images to significantly corrupt the generated images. We show that a subtle perturbation on an image can significantly impact the cross-attention layers, thus changing the mapping between text and image during the fine-tuning of customized diffusion models. Extensive experiments demonstrate that CAAT is compatible with diverse diffusion models and outperforms baseline attack methods in a more effective (more noise) and efficient (twice as fast as Anti-DreamBooth and Mist) manner.



