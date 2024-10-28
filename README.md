# Latest Adversarial Attack Papers
**update at 2024-10-28 09:35:44**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Robust Thompson Sampling Algorithms Against Reward Poisoning Attacks**

cs.LG

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19705v1) [paper-pdf](http://arxiv.org/pdf/2410.19705v1)

**Authors**: Yinglun Xu, Zhiwei Wang, Gagandeep Singh

**Abstract**: Thompson sampling is one of the most popular learning algorithms for online sequential decision-making problems and has rich real-world applications. However, current Thompson sampling algorithms are limited by the assumption that the rewards received are uncorrupted, which may not be true in real-world applications where adversarial reward poisoning exists. To make Thompson sampling more reliable, we want to make it robust against adversarial reward poisoning. The main challenge is that one can no longer compute the actual posteriors for the true reward, as the agent can only observe the rewards after corruption. In this work, we solve this problem by computing pseudo-posteriors that are less likely to be manipulated by the attack. We propose robust algorithms based on Thompson sampling for the popular stochastic and contextual linear bandit settings in both cases where the agent is aware or unaware of the budget of the attacker. We theoretically show that our algorithms guarantee near-optimal regret under any attack strategy.



## **2. A constrained optimization approach to improve robustness of neural networks**

cs.LG

29 pages, 4 figures, 5 tables

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2409.13770v2) [paper-pdf](http://arxiv.org/pdf/2409.13770v2)

**Authors**: Shudian Zhao, Jan Kronqvist

**Abstract**: In this paper, we present a novel nonlinear programming-based approach to fine-tune pre-trained neural networks to improve robustness against adversarial attacks while maintaining high accuracy on clean data. Our method introduces adversary-correction constraints to ensure correct classification of adversarial data and minimizes changes to the model parameters. We propose an efficient cutting-plane-based algorithm to iteratively solve the large-scale nonconvex optimization problem by approximating the feasible region through polyhedral cuts and balancing between robustness and accuracy. Computational experiments on standard datasets such as MNIST and CIFAR10 demonstrate that the proposed approach significantly improves robustness, even with a very small set of adversarial data, while maintaining minimal impact on accuracy.



## **3. Detecting adversarial attacks on random samples**

math.PR

title changed; introduction expanded; new results about spherical  attacks

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2408.06166v2) [paper-pdf](http://arxiv.org/pdf/2408.06166v2)

**Authors**: Gleb Smirnov

**Abstract**: This paper studies the problem of detecting adversarial perturbations in a sequence of observations. Given a data sample $X_1, \ldots, X_n$ drawn from a standard normal distribution, an adversary, after observing the sample, can perturb each observation by a fixed magnitude or leave it unchanged. We explore the relationship between the perturbation magnitude, the sparsity of the perturbation, and the detectability of the adversary's actions, establishing precise thresholds for when detection becomes impossible.



## **4. Corpus Poisoning via Approximate Greedy Gradient Descent**

cs.IR

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2406.05087v2) [paper-pdf](http://arxiv.org/pdf/2406.05087v2)

**Authors**: Jinyan Su, Preslav Nakov, Claire Cardie

**Abstract**: Dense retrievers are widely used in information retrieval and have also been successfully extended to other knowledge intensive areas such as language models, e.g., Retrieval-Augmented Generation (RAG) systems. Unfortunately, they have recently been shown to be vulnerable to corpus poisoning attacks in which a malicious user injects a small fraction of adversarial passages into the retrieval corpus to trick the system into returning these passages among the top-ranked results for a broad set of user queries. Further study is needed to understand the extent to which these attacks could limit the deployment of dense retrievers in real-world applications. In this work, we propose Approximate Greedy Gradient Descent (AGGD), a new attack on dense retrieval systems based on the widely used HotFlip method for efficiently generating adversarial passages. We demonstrate that AGGD can select a higher quality set of token-level perturbations than HotFlip by replacing its random token sampling with a more structured search. Experimentally, we show that our method achieves a high attack success rate on several datasets and using several retrievers, and can generalize to unseen queries and new domains. Notably, our method is extremely effective in attacking the ANCE retrieval model, achieving attack success rates that are 15.24\% and 17.44\% higher on the NQ and MS MARCO datasets, respectively, compared to HotFlip. Additionally, we demonstrate AGGD's potential to replace HotFlip in other adversarial attacks, such as knowledge poisoning of RAG systems.



## **5. Adversarial Attacks on Large Language Models Using Regularized Relaxation**

cs.LG

8 pages, 6 figures

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.19160v1) [paper-pdf](http://arxiv.org/pdf/2410.19160v1)

**Authors**: Samuel Jacob Chacko, Sajib Biswas, Chashi Mahiul Islam, Fatema Tabassum Liza, Xiuwen Liu

**Abstract**: As powerful Large Language Models (LLMs) are now widely used for numerous practical applications, their safety is of critical importance. While alignment techniques have significantly improved overall safety, LLMs remain vulnerable to carefully crafted adversarial inputs. Consequently, adversarial attack methods are extensively used to study and understand these vulnerabilities. However, current attack methods face significant limitations. Those relying on optimizing discrete tokens suffer from limited efficiency, while continuous optimization techniques fail to generate valid tokens from the model's vocabulary, rendering them impractical for real-world applications. In this paper, we propose a novel technique for adversarial attacks that overcomes these limitations by leveraging regularized gradients with continuous optimization methods. Our approach is two orders of magnitude faster than the state-of-the-art greedy coordinate gradient-based method, significantly improving the attack success rate on aligned language models. Moreover, it generates valid tokens, addressing a fundamental limitation of existing continuous optimization methods. We demonstrate the effectiveness of our attack on five state-of-the-art LLMs using four datasets.



## **6. Provably Robust Watermarks for Open-Source Language Models**

cs.CR

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18861v1) [paper-pdf](http://arxiv.org/pdf/2410.18861v1)

**Authors**: Miranda Christ, Sam Gunn, Tal Malkin, Mariana Raykova

**Abstract**: The recent explosion of high-quality language models has necessitated new methods for identifying AI-generated text. Watermarking is a leading solution and could prove to be an essential tool in the age of generative AI. Existing approaches embed watermarks at inference and crucially rely on the large language model (LLM) specification and parameters being secret, which makes them inapplicable to the open-source setting. In this work, we introduce the first watermarking scheme for open-source LLMs. Our scheme works by modifying the parameters of the model, but the watermark can be detected from just the outputs of the model. Perhaps surprisingly, we prove that our watermarks are unremovable under certain assumptions about the adversary's knowledge. To demonstrate the behavior of our construction under concrete parameter instantiations, we present experimental results with OPT-6.7B and OPT-1.3B. We demonstrate robustness to both token substitution and perturbation of the model parameters. We find that the stronger of these attacks, the model-perturbation attack, requires deteriorating the quality score to 0 out of 100 in order to bring the detection rate down to 50%.



## **7. Rethinking Randomized Smoothing from the Perspective of Scalability**

cs.LG

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2312.12608v2) [paper-pdf](http://arxiv.org/pdf/2312.12608v2)

**Authors**: Anupriya Kumari, Devansh Bhardwaj, Sukrit Jindal

**Abstract**: Machine learning models have demonstrated remarkable success across diverse domains but remain vulnerable to adversarial attacks. Empirical defense mechanisms often fail, as new attacks constantly emerge, rendering existing defenses obsolete, shifting the focus to certification-based defenses. Randomized smoothing has emerged as a promising technique among notable advancements. This study reviews the theoretical foundations and empirical effectiveness of randomized smoothing and its derivatives in verifying machine learning classifiers from a perspective of scalability. We provide an in-depth exploration of the fundamental concepts underlying randomized smoothing, highlighting its theoretical guarantees in certifying robustness against adversarial perturbations and discuss the challenges of existing methodologies.



## **8. GADT: Enhancing Transferable Adversarial Attacks through Gradient-guided Adversarial Data Transformation**

cs.AI

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18648v1) [paper-pdf](http://arxiv.org/pdf/2410.18648v1)

**Authors**: Yating Ma, Xiaogang Xu, Liming Fang, Zhe Liu

**Abstract**: Current Transferable Adversarial Examples (TAE) are primarily generated by adding Adversarial Noise (AN). Recent studies emphasize the importance of optimizing Data Augmentation (DA) parameters along with AN, which poses a greater threat to real-world AI applications. However, existing DA-based strategies often struggle to find optimal solutions due to the challenging DA search procedure without proper guidance. In this work, we propose a novel DA-based attack algorithm, GADT. GADT identifies suitable DA parameters through iterative antagonism and uses posterior estimates to update AN based on these parameters. We uniquely employ a differentiable DA operation library to identify adversarial DA parameters and introduce a new loss function as a metric during DA optimization. This loss term enhances adversarial effects while preserving the original image content, maintaining attack crypticity. Extensive experiments on public datasets with various networks demonstrate that GADT can be integrated with existing transferable attack methods, updating their DA parameters effectively while retaining their AN formulation strategies. Furthermore, GADT can be utilized in other black-box attack scenarios, e.g., query-based attacks, offering a new avenue to enhance attacks on real-world AI applications in both research and industrial contexts.



## **9. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

cs.CL

18 pages

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18469v1) [paper-pdf](http://arxiv.org/pdf/2410.18469v1)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99% ASR on GPT-3.5 and 49% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM



## **10. Effects of Scale on Language Model Robustness**

cs.LG

36 pages; updated to include new results and analysis

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2407.18213v3) [paper-pdf](http://arxiv.org/pdf/2407.18213v3)

**Authors**: Nikolaus Howe, Ian McKenzie, Oskar Hollinsworth, Michał Zajac, Tom Tseng, Aaron Tucker, Pierre-Luc Bacon, Adam Gleave

**Abstract**: Language models exhibit scaling laws, whereby increasing model and dataset size yields predictable decreases in negative log likelihood, unlocking a dazzling array of capabilities. This phenomenon spurs many companies to train ever larger models in pursuit of ever improved performance. Yet, these models are vulnerable to adversarial inputs such as ``jailbreaks'' and prompt injections that induce models to perform undesired behaviors, posing a growing risk as models become more capable. Prior work indicates that computer vision models become more robust with model and data scaling, raising the question: does language model robustness also improve with scale?   We study this question empirically in the classification setting, finding that without explicit defense training, larger models tend to be modestly more robust on most tasks, though the effect is not reliable. Even with the advantage conferred by scale, undefended models remain easy to attack in absolute terms, and we thus turn our attention to explicitly training models for adversarial robustness, which we show to be a much more compute-efficient defense than scaling model size alone. In this setting, we also observe that adversarially trained larger models generalize faster and better to modified attacks not seen during training when compared with smaller models. Finally, we analyze the offense/defense balance of increasing compute, finding parity in some settings and an advantage for offense in others, suggesting that adversarial training alone is not sufficient to solve robustness, even at greater model scales.



## **11. Backdoor in Seconds: Unlocking Vulnerabilities in Large Pre-trained Models via Model Editing**

cs.AI

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18267v1) [paper-pdf](http://arxiv.org/pdf/2410.18267v1)

**Authors**: Dongliang Guo, Mengxuan Hu, Zihan Guan, Junfeng Guo, Thomas Hartvigsen, Sheng Li

**Abstract**: Large pre-trained models have achieved notable success across a range of downstream tasks. However, recent research shows that a type of adversarial attack ($\textit{i.e.,}$ backdoor attack) can manipulate the behavior of machine learning models through contaminating their training dataset, posing significant threat in the real-world application of large pre-trained model, especially for those customized models. Therefore, addressing the unique challenges for exploring vulnerability of pre-trained models is of paramount importance. Through empirical studies on the capability for performing backdoor attack in large pre-trained models ($\textit{e.g.,}$ ViT), we find the following unique challenges of attacking large pre-trained models: 1) the inability to manipulate or even access large training datasets, and 2) the substantial computational resources required for training or fine-tuning these models. To address these challenges, we establish new standards for an effective and feasible backdoor attack in the context of large pre-trained models. In line with these standards, we introduce our EDT model, an \textbf{E}fficient, \textbf{D}ata-free, \textbf{T}raining-free backdoor attack method. Inspired by model editing techniques, EDT injects an editing-based lightweight codebook into the backdoor of large pre-trained models, which replaces the embedding of the poisoned image with the target image without poisoning the training dataset or training the victim model. Our experiments, conducted across various pre-trained models such as ViT, CLIP, BLIP, and stable diffusion, and on downstream tasks including image classification, image captioning, and image generation, demonstrate the effectiveness of our method. Our code is available in the supplementary material.



## **12. Advancing NLP Security by Leveraging LLMs as Adversarial Engines**

cs.AI

5 pages

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18215v1) [paper-pdf](http://arxiv.org/pdf/2410.18215v1)

**Authors**: Sudarshan Srinivasan, Maria Mahbub, Amir Sadovnik

**Abstract**: This position paper proposes a novel approach to advancing NLP security by leveraging Large Language Models (LLMs) as engines for generating diverse adversarial attacks. Building upon recent work demonstrating LLMs' effectiveness in creating word-level adversarial examples, we argue for expanding this concept to encompass a broader range of attack types, including adversarial patches, universal perturbations, and targeted attacks. We posit that LLMs' sophisticated language understanding and generation capabilities can produce more effective, semantically coherent, and human-like adversarial examples across various domains and classifier architectures. This paradigm shift in adversarial NLP has far-reaching implications, potentially enhancing model robustness, uncovering new vulnerabilities, and driving innovation in defense mechanisms. By exploring this new frontier, we aim to contribute to the development of more secure, reliable, and trustworthy NLP systems for critical applications.



## **13. Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks**

cs.CL

14 pages, 6 figures, 7 tables

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18210v1) [paper-pdf](http://arxiv.org/pdf/2410.18210v1)

**Authors**: Samuele Poppi, Zheng-Xin Yong, Yifei He, Bobbie Chern, Han Zhao, Aobo Yang, Jianfeng Chi

**Abstract**: Recent advancements in Large Language Models (LLMs) have sparked widespread concerns about their safety. Recent work demonstrates that safety alignment of LLMs can be easily removed by fine-tuning with a few adversarially chosen instruction-following examples, i.e., fine-tuning attacks. We take a further step to understand fine-tuning attacks in multilingual LLMs. We first discover cross-lingual generalization of fine-tuning attacks: using a few adversarially chosen instruction-following examples in one language, multilingual LLMs can also be easily compromised (e.g., multilingual LLMs fail to refuse harmful prompts in other languages). Motivated by this finding, we hypothesize that safety-related information is language-agnostic and propose a new method termed Safety Information Localization (SIL) to identify the safety-related information in the model parameter space. Through SIL, we validate this hypothesis and find that only changing 20% of weight parameters in fine-tuning attacks can break safety alignment across all languages. Furthermore, we provide evidence to the alternative pathways hypothesis for why freezing safety-related parameters does not prevent fine-tuning attacks, and we demonstrate that our attack vector can still jailbreak LLMs adapted to new languages.



## **14. Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models**

cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.02916v2) [paper-pdf](http://arxiv.org/pdf/2410.02916v2)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern of large language models (LLMs) in their open deployment. To this end, safeguard methods aim to enforce the ethical and responsible use of LLMs through safety alignment or guardrail mechanisms. However, we found that the malicious attackers could exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a new denial-of-service (DoS) attack on LLMs. Specifically, by software or phishing attacks on user client software, attackers insert a short, seemingly innocuous adversarial prompt into to user prompt templates in configuration files; thus, this prompt appears in final user requests without visibility in the user interface and is not trivial to identify. By designing an optimization process that utilizes gradient and attention information, our attack can automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97\% of user requests on Llama Guard 3. The attack presents a new dimension of evaluating LLM safeguards focusing on false positives, fundamentally different from the classic jailbreak.



## **15. Exploring the Adversarial Robustness of CLIP for AI-generated Image Detection**

cs.CV

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2407.19553v2) [paper-pdf](http://arxiv.org/pdf/2407.19553v2)

**Authors**: Vincenzo De Rosa, Fabrizio Guillaro, Giovanni Poggi, Davide Cozzolino, Luisa Verdoliva

**Abstract**: In recent years, many forensic detectors have been proposed to detect AI-generated images and prevent their use for malicious purposes. Convolutional neural networks (CNNs) have long been the dominant architecture in this field and have been the subject of intense study. However, recently proposed Transformer-based detectors have been shown to match or even outperform CNN-based detectors, especially in terms of generalization. In this paper, we study the adversarial robustness of AI-generated image detectors, focusing on Contrastive Language-Image Pretraining (CLIP)-based methods that rely on Visual Transformer (ViT) backbones and comparing their performance with CNN-based methods. We study the robustness to different adversarial attacks under a variety of conditions and analyze both numerical results and frequency-domain patterns. CLIP-based detectors are found to be vulnerable to white-box attacks just like CNN-based detectors. However, attacks do not easily transfer between CNN-based and CLIP-based methods. This is also confirmed by the different distribution of the adversarial noise patterns in the frequency domain. Overall, this analysis provides new insights into the properties of forensic detectors that can help to develop more effective strategies.



## **16. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.02240v4) [paper-pdf](http://arxiv.org/pdf/2410.02240v4)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.



## **17. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.17910v1) [paper-pdf](http://arxiv.org/pdf/2410.17910v1)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zijian Zhang, Zhengzi Xu, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.



## **18. Gradient-based Jailbreak Images for Multimodal Fusion Models**

cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.03489v2) [paper-pdf](http://arxiv.org/pdf/2410.03489v2)

**Authors**: Javier Rando, Hannah Korevaar, Erik Brinkman, Ivan Evtimov, Florian Tramèr

**Abstract**: Augmenting language models with image inputs may enable more effective jailbreak attacks through continuous optimization, unlike text inputs that require discrete optimization. However, new multimodal fusion models tokenize all input modalities using non-differentiable functions, which hinders straightforward attacks. In this work, we introduce the notion of a tokenizer shortcut that approximates tokenization with a continuous function and enables continuous optimization. We use tokenizer shortcuts to create the first end-to-end gradient image attacks against multimodal fusion models. We evaluate our attacks on Chameleon models and obtain jailbreak images that elicit harmful information for 72.5% of prompts. Jailbreak images outperform text jailbreaks optimized with the same objective and require 3x lower compute budget to optimize 50x more input tokens. Finally, we find that representation engineering defenses, like Circuit Breakers, trained only on text attacks can effectively transfer to adversarial image inputs.



## **19. STBA: Towards Evaluating the Robustness of DNNs for Query-Limited Black-box Scenario**

cs.CV

Accepted by T-MM

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2404.00362v2) [paper-pdf](http://arxiv.org/pdf/2404.00362v2)

**Authors**: Renyang Liu, Kwok-Yan Lam, Wei Zhou, Sixing Wu, Jun Zhao, Dongting Hu, Mingming Gong

**Abstract**: Many attack techniques have been proposed to explore the vulnerability of DNNs and further help to improve their robustness. Despite the significant progress made recently, existing black-box attack methods still suffer from unsatisfactory performance due to the vast number of queries needed to optimize desired perturbations. Besides, the other critical challenge is that adversarial examples built in a noise-adding manner are abnormal and struggle to successfully attack robust models, whose robustness is enhanced by adversarial training against small perturbations. There is no doubt that these two issues mentioned above will significantly increase the risk of exposure and result in a failure to dig deeply into the vulnerability of DNNs. Hence, it is necessary to evaluate DNNs' fragility sufficiently under query-limited settings in a non-additional way. In this paper, we propose the Spatial Transform Black-box Attack (STBA), a novel framework to craft formidable adversarial examples in the query-limited scenario. Specifically, STBA introduces a flow field to the high-frequency part of clean images to generate adversarial examples and adopts the following two processes to enhance their naturalness and significantly improve the query efficiency: a) we apply an estimated flow field to the high-frequency part of clean images to generate adversarial examples instead of introducing external noise to the benign image, and b) we leverage an efficient gradient estimation method based on a batch of samples to optimize such an ideal flow field under query-limited settings. Compared to existing score-based black-box baselines, extensive experiments indicated that STBA could effectively improve the imperceptibility of the adversarial examples and remarkably boost the attack success rate under query-limited settings.



## **20. DIP-Watermark: A Double Identity Protection Method Based on Robust Adversarial Watermark**

cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2404.14693v2) [paper-pdf](http://arxiv.org/pdf/2404.14693v2)

**Authors**: Yunming Zhang, Dengpan Ye, Caiyun Xie, Sipeng Shen, Ziyi Liu, Jiacheng Deng, Long Tang

**Abstract**: The wide deployment of Face Recognition (FR) systems poses privacy risks. One countermeasure is adversarial attack, deceiving unauthorized malicious FR, but it also disrupts regular identity verification of trusted authorizers, exacerbating the potential threat of identity impersonation. To address this, we propose the first double identity protection scheme based on traceable adversarial watermarking, termed DIP-Watermark. DIP-Watermark employs a one-time watermark embedding to deceive unauthorized FR models and allows authorizers to perform identity verification by extracting the watermark. Specifically, we propose an information-guided adversarial attack against FR models. The encoder embeds an identity-specific watermark into the deep feature space of the carrier, guiding recognizable features of the image to deviate from the source identity. We further adopt a collaborative meta-optimization strategy compatible with sub-tasks, which regularizes the joint optimization direction of the encoder and decoder. This strategy enhances the representation of universal carrier features, mitigating multi-objective optimization conflicts in watermarking. Experiments confirm that DIP-Watermark achieves significant attack success rates and traceability accuracy on state-of-the-art FR models, exhibiting remarkable robustness that outperforms the existing privacy protection methods using adversarial attacks and deep watermarking, or simple combinations of the two. Our work potentially opens up new insights into proactive protection for FR privacy.



## **21. IBGP: Imperfect Byzantine Generals Problem for Zero-Shot Robustness in Communicative Multi-Agent Systems**

cs.MA

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.16237v2) [paper-pdf](http://arxiv.org/pdf/2410.16237v2)

**Authors**: Yihuan Mao, Yipeng Kang, Peilun Li, Ning Zhang, Wei Xu, Chongjie Zhang

**Abstract**: As large language model (LLM) agents increasingly integrate into our infrastructure, their robust coordination and message synchronization become vital. The Byzantine Generals Problem (BGP) is a critical model for constructing resilient multi-agent systems (MAS) under adversarial attacks. It describes a scenario where malicious agents with unknown identities exist in the system-situations that, in our context, could result from LLM agents' hallucinations or external attacks. In BGP, the objective of the entire system is to reach a consensus on the action to be taken. Traditional BGP requires global consensus among all agents; however, in practical scenarios, global consensus is not always necessary and can even be inefficient. Therefore, there is a pressing need to explore a refined version of BGP that aligns with the local coordination patterns observed in MAS. We refer to this refined version as Imperfect BGP (IBGP) in our research, aiming to address this discrepancy. To tackle this issue, we propose a framework that leverages consensus protocols within general MAS settings, providing provable resilience against communication attacks and adaptability to changing environments, as validated by empirical results. Additionally, we present a case study in a sensor network environment to illustrate the practical application of our protocol.



## **22. The Ultimate Combo: Boosting Adversarial Example Transferability by Composing Data Augmentations**

cs.CV

Accepted by AISec'24

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2312.11309v2) [paper-pdf](http://arxiv.org/pdf/2312.11309v2)

**Authors**: Zebin Yun, Achi-Or Weingarten, Eyal Ronen, Mahmood Sharif

**Abstract**: To help adversarial examples generalize from surrogate machine-learning (ML) models to targets, certain transferability-based black-box evasion attacks incorporate data augmentations (e.g., random resizing). Yet, prior work has explored limited augmentations and their composition. To fill the gap, we systematically studied how data augmentation affects transferability. Specifically, we explored 46 augmentation techniques originally proposed to help ML models generalize to unseen benign samples, and assessed how they impact transferability, when applied individually or composed. Performing exhaustive search on a small subset of augmentation techniques and genetic search on all techniques, we identified augmentation combinations that help promote transferability. Extensive experiments with the ImageNet and CIFAR-10 datasets and 18 models showed that simple color-space augmentations (e.g., color to greyscale) attain high transferability when combined with standard augmentations. Furthermore, we discovered that composing augmentations impacts transferability mostly monotonically (i.e., more augmentations $\rightarrow$ $\ge$transferability). We also found that the best composition significantly outperformed the state of the art (e.g., 91.8% vs. $\le$82.5% average transferability to adversarially trained targets on ImageNet). Lastly, our theoretical analysis, backed by empirical evidence, intuitively explains why certain augmentations promote transferability.



## **23. Diffusion Models are Certifiably Robust Classifiers**

cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2402.02316v3) [paper-pdf](http://arxiv.org/pdf/2402.02316v3)

**Authors**: Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: Generative learning, recognized for its effective modeling of data distributions, offers inherent advantages in handling out-of-distribution instances, especially for enhancing robustness to adversarial attacks. Among these, diffusion classifiers, utilizing powerful diffusion models, have demonstrated superior empirical robustness. However, a comprehensive theoretical understanding of their robustness is still lacking, raising concerns about their vulnerability to stronger future attacks. In this study, we prove that diffusion classifiers possess $O(1)$ Lipschitzness, and establish their certified robustness, demonstrating their inherent resilience. To achieve non-constant Lipschitzness, thereby obtaining much tighter certified robustness, we generalize diffusion classifiers to classify Gaussian-corrupted data. This involves deriving the evidence lower bounds (ELBOs) for these distributions, approximating the likelihood using the ELBO, and calculating classification probabilities via Bayes' theorem. Experimental results show the superior certified robustness of these Noised Diffusion Classifiers (NDCs). Notably, we achieve over 80% and 70% certified robustness on CIFAR-10 under adversarial perturbations with \(\ell_2\) norms less than 0.25 and 0.5, respectively, using a single off-the-shelf diffusion model without any additional data.



## **24. A provable initialization and robust clustering method for general mixture models**

math.ST

51 pages, corrected typos, updated structures and results are  improved

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2401.05574v3) [paper-pdf](http://arxiv.org/pdf/2401.05574v3)

**Authors**: Soham Jana, Jianqing Fan, Sanjeev Kulkarni

**Abstract**: Clustering is a fundamental tool in statistical machine learning in the presence of heterogeneous data. Most recent results focus primarily on optimal mislabeling guarantees when data are distributed around centroids with sub-Gaussian errors. Yet, the restrictive sub-Gaussian model is often invalid in practice since various real-world applications exhibit heavy tail distributions around the centroids or suffer from possible adversarial attacks that call for robust clustering with a robust data-driven initialization. In this paper, we present initialization and subsequent clustering methods that provably guarantee near-optimal mislabeling for general mixture models when the number of clusters and data dimensions are finite. We first introduce a hybrid clustering technique with a novel multivariate trimmed mean type centroid estimate to produce mislabeling guarantees under a weak initialization condition for general error distributions around the centroids. A matching lower bound is derived, up to factors depending on the number of clusters. In addition, our approach also produces similar mislabeling guarantees even in the presence of adversarial outliers. Our results reduce to the sub-Gaussian case in finite dimensions when errors follow sub-Gaussian distributions. To solve the problem thoroughly, we also present novel data-driven robust initialization techniques and show that, with probabilities approaching one, these initial centroid estimates are sufficiently good for the subsequent clustering algorithm to achieve the optimal mislabeling rates. Furthermore, we demonstrate that the Lloyd algorithm is suboptimal for more than two clusters even when errors are Gaussian and for two clusters when error distributions have heavy tails. Both simulated data and real data examples further support our robust initialization procedure and clustering algorithm.



## **25. Detecting Adversarial Examples**

cs.LG

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17442v1) [paper-pdf](http://arxiv.org/pdf/2410.17442v1)

**Authors**: Furkan Mumcu, Yasin Yilmaz

**Abstract**: Deep Neural Networks (DNNs) have been shown to be vulnerable to adversarial examples. While numerous successful adversarial attacks have been proposed, defenses against these attacks remain relatively understudied. Existing defense approaches either focus on negating the effects of perturbations caused by the attacks to restore the DNNs' original predictions or use a secondary model to detect adversarial examples. However, these methods often become ineffective due to the continuous advancements in attack techniques. We propose a novel universal and lightweight method to detect adversarial examples by analyzing the layer outputs of DNNs. Through theoretical justification and extensive experiments, we demonstrate that our detection method is highly effective, compatible with any DNN architecture, and applicable across different domains, such as image, video, and audio.



## **26. Meta Stackelberg Game: Robust Federated Learning against Adaptive and Mixed Poisoning Attacks**

cs.LG

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17431v1) [paper-pdf](http://arxiv.org/pdf/2410.17431v1)

**Authors**: Tao Li, Henger Li, Yunian Pan, Tianyi Xu, Zizhan Zheng, Quanyan Zhu

**Abstract**: Federated learning (FL) is susceptible to a range of security threats. Although various defense mechanisms have been proposed, they are typically non-adaptive and tailored to specific types of attacks, leaving them insufficient in the face of multiple uncertain, unknown, and adaptive attacks employing diverse strategies. This work formulates adversarial federated learning under a mixture of various attacks as a Bayesian Stackelberg Markov game, based on which we propose the meta-Stackelberg defense composed of pre-training and online adaptation. {The gist is to simulate strong attack behavior using reinforcement learning (RL-based attacks) in pre-training and then design meta-RL-based defense to combat diverse and adaptive attacks.} We develop an efficient meta-learning approach to solve the game, leading to a robust and adaptive FL defense. Theoretically, our meta-learning algorithm, meta-Stackelberg learning, provably converges to the first-order $\varepsilon$-meta-equilibrium point in $O(\varepsilon^{-2})$ gradient iterations with $O(\varepsilon^{-4})$ samples per iteration. Experiments show that our meta-Stackelberg framework performs superbly against strong model poisoning and backdoor attacks of uncertain and unknown types.



## **27. AdvWeb: Controllable Black-box Attacks on VLM-powered Web Agents**

cs.CR

15 pages

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17401v1) [paper-pdf](http://arxiv.org/pdf/2410.17401v1)

**Authors**: Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, Bo Li

**Abstract**: Vision Language Models (VLMs) have revolutionized the creation of generalist web agents, empowering them to autonomously complete diverse tasks on real-world websites, thereby boosting human efficiency and productivity. However, despite their remarkable capabilities, the safety and security of these agents against malicious attacks remain critically underexplored, raising significant concerns about their safe deployment. To uncover and exploit such vulnerabilities in web agents, we provide AdvWeb, a novel black-box attack framework designed against web agents. AdvWeb trains an adversarial prompter model that generates and injects adversarial prompts into web pages, misleading web agents into executing targeted adversarial actions such as inappropriate stock purchases or incorrect bank transactions, actions that could lead to severe real-world consequences. With only black-box access to the web agent, we train and optimize the adversarial prompter model using DPO, leveraging both successful and failed attack strings against the target agent. Unlike prior approaches, our adversarial string injection maintains stealth and control: (1) the appearance of the website remains unchanged before and after the attack, making it nearly impossible for users to detect tampering, and (2) attackers can modify specific substrings within the generated adversarial string to seamlessly change the attack objective (e.g., purchasing stocks from a different company), enhancing attack flexibility and efficiency. We conduct extensive evaluations, demonstrating that AdvWeb achieves high success rates in attacking SOTA GPT-4V-based VLM agent across various web tasks. Our findings expose critical vulnerabilities in current LLM/VLM-based agents, emphasizing the urgent need for developing more reliable web agents and effective defenses. Our code and data are available at https://ai-secure.github.io/AdvWeb/ .



## **28. Learning to Poison Large Language Models During Instruction Tuning**

cs.LG

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2402.13459v2) [paper-pdf](http://arxiv.org/pdf/2402.13459v2)

**Authors**: Yao Qiang, Xiangyu Zhou, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during the instruction tuning of LLMs and emphasizes the necessity of safeguarding LLMs against data poisoning attacks.



## **29. Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods**

cs.CL

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17222v1) [paper-pdf](http://arxiv.org/pdf/2410.17222v1)

**Authors**: Tsachi Blau, Moshe Kimhi, Yonatan Belinkov, Alexander Bronstein, Chaim Baskin

**Abstract**: Fine-tuning Large Language Models (LLMs) typically involves updating at least a few billions of parameters. A more parameter-efficient approach is Prompt Tuning (PT), which updates only a few learnable tokens, and differently, In-Context Learning (ICL) adapts the model to a new task by simply including examples in the input without any training. When applying optimization-based methods, such as fine-tuning and PT for few-shot learning, the model is specifically adapted to the small set of training examples, whereas ICL leaves the model unchanged. This distinction makes traditional learning methods more prone to overfitting; in contrast, ICL is less sensitive to the few-shot scenario. While ICL is not prone to overfitting, it does not fully extract the information that exists in the training examples. This work introduces Context-aware Prompt Tuning (CPT), a method inspired by ICL, PT, and adversarial attacks. We build on the ICL strategy of concatenating examples before the input, but we extend this by PT-like learning, refining the context embedding through iterative optimization to extract deeper insights from the training examples. We carefully modify specific context tokens, considering the unique structure of input and output formats. Inspired by adversarial attacks, we adjust the input based on the labels present in the context, focusing on minimizing, rather than maximizing, the loss. Moreover, we apply a projected gradient descent algorithm to keep token embeddings close to their original values, under the assumption that the user-provided data is inherently valuable. Our method has been shown to achieve superior accuracy across multiple classification tasks using various LLM models.



## **30. Remote Timing Attacks on Efficient Language Model Inference**

cs.CR

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17175v1) [paper-pdf](http://arxiv.org/pdf/2410.17175v1)

**Authors**: Nicholas Carlini, Milad Nasr

**Abstract**: Scaling up language models has significantly increased their capabilities. But larger models are slower models, and so there is now an extensive body of work (e.g., speculative sampling or parallel decoding) that improves the (average case) efficiency of language model generation. But these techniques introduce data-dependent timing characteristics. We show it is possible to exploit these timing differences to mount a timing attack. By monitoring the (encrypted) network traffic between a victim user and a remote language model, we can learn information about the content of messages by noting when responses are faster or slower. With complete black-box access, on open source systems we show how it is possible to learn the topic of a user's conversation (e.g., medical advice vs. coding assistance) with 90%+ precision, and on production systems like OpenAI's ChatGPT and Anthropic's Claude we can distinguish between specific messages or infer the user's language. We further show that an active adversary can leverage a boosting attack to recover PII placed in messages (e.g., phone numbers or credit card numbers) for open source systems. We conclude with potential defenses and directions for future work.



## **31. FDINet: Protecting against DNN Model Extraction via Feature Distortion Index**

cs.CR

Accepted to IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2306.11338v3) [paper-pdf](http://arxiv.org/pdf/2306.11338v3)

**Authors**: Hongwei Yao, Zheng Li, Haiqin Weng, Feng Xue, Zhan Qin, Kui Ren

**Abstract**: Machine Learning as a Service (MLaaS) platforms have gained popularity due to their accessibility, cost-efficiency, scalability, and rapid development capabilities. However, recent research has highlighted the vulnerability of cloud-based models in MLaaS to model extraction attacks. In this paper, we introduce FDINET, a novel defense mechanism that leverages the feature distribution of deep neural network (DNN) models. Concretely, by analyzing the feature distribution from the adversary's queries, we reveal that the feature distribution of these queries deviates from that of the model's training set. Based on this key observation, we propose Feature Distortion Index (FDI), a metric designed to quantitatively measure the feature distribution deviation of received queries. The proposed FDINET utilizes FDI to train a binary detector and exploits FDI similarity to identify colluding adversaries from distributed extraction attacks. We conduct extensive experiments to evaluate FDINET against six state-of-the-art extraction attacks on four benchmark datasets and four popular model architectures. Empirical results demonstrate the following findings FDINET proves to be highly effective in detecting model extraction, achieving a 100% detection accuracy on DFME and DaST. FDINET is highly efficient, using just 50 queries to raise an extraction alarm with an average confidence of 96.08% for GTSRB. FDINET exhibits the capability to identify colluding adversaries with an accuracy exceeding 91%. Additionally, it demonstrates the ability to detect two types of adaptive attacks.



## **32. Adversarial Challenges in Network Intrusion Detection Systems: Research Insights and Future Prospects**

cs.CR

35 pages

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2409.18736v3) [paper-pdf](http://arxiv.org/pdf/2409.18736v3)

**Authors**: Sabrine Ennaji, Fabio De Gaspari, Dorjan Hitaj, Alicia Kbidi, Luigi V. Mancini

**Abstract**: Machine learning has brought significant advances in cybersecurity, particularly in the development of Intrusion Detection Systems (IDS). These improvements are mainly attributed to the ability of machine learning algorithms to identify complex relationships between features and effectively generalize to unseen data. Deep neural networks, in particular, contributed to this progress by enabling the analysis of large amounts of training data, significantly enhancing detection performance. However, machine learning models remain vulnerable to adversarial attacks, where carefully crafted input data can mislead the model into making incorrect predictions. While adversarial threats in unstructured data, such as images and text, have been extensively studied, their impact on structured data like network traffic is less explored. This survey aims to address this gap by providing a comprehensive review of machine learning-based Network Intrusion Detection Systems (NIDS) and thoroughly analyzing their susceptibility to adversarial attacks. We critically examine existing research in NIDS, highlighting key trends, strengths, and limitations, while identifying areas that require further exploration. Additionally, we discuss emerging challenges in the field and offer insights for the development of more robust and resilient NIDS. In summary, this paper enhances the understanding of adversarial attacks and defenses in NIDS and guide future research in improving the robustness of machine learning models in cybersecurity applications.



## **33. A Self-Organizing Clustering System for Unsupervised Distribution Shift Detection**

cs.LG

Revised version of the accepted manuscript to IJCNN'2024. Main  corrections were in Section 2.2 and Section 3.3. In Section 2.2 was corrected  expression (3), and in Section 3.3 in the definition of the elements of the  matrix $D$ it was a typo where $\phi(x)$ was written instead of $x$

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2404.16656v2) [paper-pdf](http://arxiv.org/pdf/2404.16656v2)

**Authors**: Sebastián Basterrech, Line Clemmensen, Gerardo Rubino

**Abstract**: Modeling non-stationary data is a challenging problem in the field of continual learning, and data distribution shifts may result in negative consequences on the performance of a machine learning model. Classic learning tools are often vulnerable to perturbations of the input covariates, and are sensitive to outliers and noise, and some tools are based on rigid algebraic assumptions. Distribution shifts are frequently occurring due to changes in raw materials for production, seasonality, a different user base, or even adversarial attacks. Therefore, there is a need for more effective distribution shift detection techniques. In this work, we propose a continual learning framework for monitoring and detecting distribution changes. We explore the problem in a latent space generated by a bio-inspired self-organizing clustering and statistical aspects of the latent space. In particular, we investigate the projections made by two topology-preserving maps: the Self-Organizing Map and the Scale Invariant Map. Our method can be applied in both a supervised and an unsupervised context. We construct the assessment of changes in the data distribution as a comparison of Gaussian signals, making the proposed method fast and robust. We compare it to other unsupervised techniques, specifically Principal Component Analysis (PCA) and Kernel-PCA. Our comparison involves conducting experiments using sequences of images (based on MNIST and injected shifts with adversarial samples), chemical sensor measurements, and the environmental variable related to ozone levels. The empirical study reveals the potential of the proposed approach.



## **34. Test-time Adversarial Defense with Opposite Adversarial Path and High Attack Time Cost**

cs.LG

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.16805v1) [paper-pdf](http://arxiv.org/pdf/2410.16805v1)

**Authors**: Cheng-Han Yeh, Kuanchun Yu, Chun-Shien Lu

**Abstract**: Deep learning models are known to be vulnerable to adversarial attacks by injecting sophisticated designed perturbations to input data. Training-time defenses still exhibit a significant performance gap between natural accuracy and robust accuracy. In this paper, we investigate a new test-time adversarial defense method via diffusion-based recovery along opposite adversarial paths (OAPs). We present a purifier that can be plugged into a pre-trained model to resist adversarial attacks. Different from prior arts, the key idea is excessive denoising or purification by integrating the opposite adversarial direction with reverse diffusion to push the input image further toward the opposite adversarial direction. For the first time, we also exemplify the pitfall of conducting AutoAttack (Rand) for diffusion-based defense methods. Through the lens of time complexity, we examine the trade-off between the effectiveness of adaptive attack and its computation complexity against our defense. Experimental evaluation along with time cost analysis verifies the effectiveness of the proposed method.



## **35. Evaluating the Effectiveness of Attack-Agnostic Features for Morphing Attack Detection**

cs.CV

Published in the 2024 IEEE International Joint Conference on  Biometrics (IJCB)

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.16802v1) [paper-pdf](http://arxiv.org/pdf/2410.16802v1)

**Authors**: Laurent Colbois, Sébastien Marcel

**Abstract**: Morphing attacks have diversified significantly over the past years, with new methods based on generative adversarial networks (GANs) and diffusion models posing substantial threats to face recognition systems. Recent research has demonstrated the effectiveness of features extracted from large vision models pretrained on bonafide data only (attack-agnostic features) for detecting deep generative images. Building on this, we investigate the potential of these image representations for morphing attack detection (MAD). We develop supervised detectors by training a simple binary linear SVM on the extracted features and one-class detectors by modeling the distribution of bonafide features with a Gaussian Mixture Model (GMM). Our method is evaluated across a comprehensive set of attacks and various scenarios, including generalization to unseen attacks, different source datasets, and print-scan data. Our results indicate that attack-agnostic features can effectively detect morphing attacks, outperforming traditional supervised and one-class detectors from the literature in most scenarios. Additionally, we provide insights into the strengths and limitations of each considered representation and discuss potential future research directions to further enhance the robustness and generalizability of our approach.



## **36. Imprompter: Tricking LLM Agents into Improper Tool Use**

cs.CR

website: https://imprompter.ai code:  https://github.com/Reapor-Yurnero/imprompter v2 changelog: add new results to  Table 3, correct several typos

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.14923v2) [paper-pdf](http://arxiv.org/pdf/2410.14923v2)

**Authors**: Xiaohan Fu, Shuheng Li, Zihan Wang, Yihao Liu, Rajesh K. Gupta, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Model (LLM) Agents are an emerging computing paradigm that blends generative machine learning with tools such as code interpreters, web browsing, email, and more generally, external resources. These agent-based systems represent an emerging shift in personal computing. We contribute to the security foundations of agent-based systems and surface a new class of automatically computed obfuscated adversarial prompt attacks that violate the confidentiality and integrity of user resources connected to an LLM agent. We show how prompt optimization techniques can find such prompts automatically given the weights of a model. We demonstrate that such attacks transfer to production-level agents. For example, we show an information exfiltration attack on Mistral's LeChat agent that analyzes a user's conversation, picks out personally identifiable information, and formats it into a valid markdown command that results in leaking that data to the attacker's server. This attack shows a nearly 80% success rate in an end-to-end evaluation. We conduct a range of experiments to characterize the efficacy of these attacks and find that they reliably work on emerging agent-based systems like Mistral's LeChat, ChatGLM, and Meta's Llama. These attacks are multimodal, and we show variants in the text-only and image domains.



## **37. (Quantum) Indifferentiability and Pre-Computation**

quant-ph

24 pages

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.16595v1) [paper-pdf](http://arxiv.org/pdf/2410.16595v1)

**Authors**: Joseph Carolan, Alexander Poremba, Mark Zhandry

**Abstract**: Indifferentiability is a popular cryptographic paradigm for analyzing the security of ideal objects -- both in a classical as well as in a quantum world. It is typically stated in the form of a composable and simulation-based definition, and captures what it means for a construction (e.g., a cryptographic hash function) to be ``as good as'' an ideal object (e.g., a random oracle). Despite its strength, indifferentiability is not known to offer security against pre-processing attacks in which the adversary gains access to (classical or quantum) advice that is relevant to the particular construction. In this work, we show that indifferentiability is (generically) insufficient for capturing pre-computation. To accommodate this shortcoming, we propose a strengthening of indifferentiability which is not only composable but also takes arbitrary pre-computation into account. As an application, we show that the one-round sponge is indifferentiable (with pre-computation) from a random oracle. This yields the first (and tight) classical/quantum space-time trade-off for one-round sponge inversion.



## **38. Conflict-Aware Adversarial Training**

cs.LG

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16579v1) [paper-pdf](http://arxiv.org/pdf/2410.16579v1)

**Authors**: Zhiyu Xue, Haohan Wang, Yao Qin, Ramtin Pedarsani

**Abstract**: Adversarial training is the most effective method to obtain adversarial robustness for deep neural networks by directly involving adversarial samples in the training procedure. To obtain an accurate and robust model, the weighted-average method is applied to optimize standard loss and adversarial loss simultaneously. In this paper, we argue that the weighted-average method does not provide the best tradeoff for the standard performance and adversarial robustness. We argue that the failure of the weighted-average method is due to the conflict between the gradients derived from standard and adversarial loss, and further demonstrate such a conflict increases with attack budget theoretically and practically. To alleviate this problem, we propose a new trade-off paradigm for adversarial training with a conflict-aware factor for the convex combination of standard and adversarial loss, named \textbf{Conflict-Aware Adversarial Training~(CA-AT)}. Comprehensive experimental results show that CA-AT consistently offers a superior trade-off between standard performance and adversarial robustness under the settings of adversarial training from scratch and parameter-efficient finetuning.



## **39. SleeperNets: Universal Backdoor Poisoning Attacks Against Reinforcement Learning Agents**

cs.LG

23 pages, 14 figures, NeurIPS

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2405.20539v2) [paper-pdf](http://arxiv.org/pdf/2405.20539v2)

**Authors**: Ethan Rathbun, Christopher Amato, Alina Oprea

**Abstract**: Reinforcement learning (RL) is an actively growing field that is seeing increased usage in real-world, safety-critical applications -- making it paramount to ensure the robustness of RL algorithms against adversarial attacks. In this work we explore a particularly stealthy form of training-time attacks against RL -- backdoor poisoning. Here the adversary intercepts the training of an RL agent with the goal of reliably inducing a particular action when the agent observes a pre-determined trigger at inference time. We uncover theoretical limitations of prior work by proving their inability to generalize across domains and MDPs. Motivated by this, we formulate a novel poisoning attack framework which interlinks the adversary's objectives with those of finding an optimal policy -- guaranteeing attack success in the limit. Using insights from our theoretical analysis we develop ``SleeperNets'' as a universal backdoor attack which exploits a newly proposed threat model and leverages dynamic reward poisoning techniques. We evaluate our attack in 6 environments spanning multiple domains and demonstrate significant improvements in attack success over existing methods, while preserving benign episodic return.



## **40. Adversarial Inception for Bounded Backdoor Poisoning in Deep Reinforcement Learning**

cs.LG

10 pages, 5 figures, ICLR 2025

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.13995v2) [paper-pdf](http://arxiv.org/pdf/2410.13995v2)

**Authors**: Ethan Rathbun, Christopher Amato, Alina Oprea

**Abstract**: Recent works have demonstrated the vulnerability of Deep Reinforcement Learning (DRL) algorithms against training-time, backdoor poisoning attacks. These attacks induce pre-determined, adversarial behavior in the agent upon observing a fixed trigger during deployment while allowing the agent to solve its intended task during training. Prior attacks rely on arbitrarily large perturbations to the agent's rewards to achieve both of these objectives - leaving them open to detection. Thus, in this work, we propose a new class of backdoor attacks against DRL which achieve state of the art performance while minimally altering the agent's rewards. These "inception" attacks train the agent to associate the targeted adversarial behavior with high returns by inducing a disjunction between the agent's chosen action and the true action executed in the environment during training. We formally define these attacks and prove they can achieve both adversarial objectives. We then devise an online inception attack which significantly out-performs prior attacks under bounded reward constraints.



## **41. A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns**

cs.CL

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16155v1) [paper-pdf](http://arxiv.org/pdf/2410.16155v1)

**Authors**: Tianyi Men, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: With the development of large language models, they are widely used as agents in various fields. A key component of agents is memory, which stores vital information but is susceptible to jailbreak attacks. Existing research mainly focuses on single-agent attacks and shared memory attacks. However, real-world scenarios often involve independent memory. In this paper, we propose the Troublemaker Makes Chaos in Honest Town (TMCHT) task, a large-scale, multi-agent, multi-topology text-based attack evaluation framework. TMCHT involves one attacker agent attempting to mislead an entire society of agents. We identify two major challenges in multi-agent attacks: (1) Non-complete graph structure, (2) Large-scale systems. We attribute these challenges to a phenomenon we term toxicity disappearing. To address these issues, we propose an Adversarial Replication Contagious Jailbreak (ARCJ) method, which optimizes the retrieval suffix to make poisoned samples more easily retrieved and optimizes the replication suffix to make poisoned samples have contagious ability. We demonstrate the superiority of our approach in TMCHT, with 23.51%, 18.95%, and 52.93% improvements in line topology, star topology, and 100-agent settings. Encourage community attention to the security of multi-agent systems.



## **42. On the Geometry of Regularization in Adversarial Training: High-Dimensional Asymptotics and Generalization Bounds**

stat.ML

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16073v1) [paper-pdf](http://arxiv.org/pdf/2410.16073v1)

**Authors**: Matteo Vilucchio, Nikolaos Tsilivis, Bruno Loureiro, Julia Kempe

**Abstract**: Regularization, whether explicit in terms of a penalty in the loss or implicit in the choice of algorithm, is a cornerstone of modern machine learning. Indeed, controlling the complexity of the model class is particularly important when data is scarce, noisy or contaminated, as it translates a statistical belief on the underlying structure of the data. This work investigates the question of how to choose the regularization norm $\lVert \cdot \rVert$ in the context of high-dimensional adversarial training for binary classification. To this end, we first derive an exact asymptotic description of the robust, regularized empirical risk minimizer for various types of adversarial attacks and regularization norms (including non-$\ell_p$ norms). We complement this analysis with a uniform convergence analysis, deriving bounds on the Rademacher Complexity for this class of problems. Leveraging our theoretical results, we quantitatively characterize the relationship between perturbation size and the optimal choice of $\lVert \cdot \rVert$, confirming the intuition that, in the data scarce regime, the type of regularization becomes increasingly important for adversarial training as perturbations grow in size.



## **43. A Differentially Private Energy Trading Mechanism Approaching Social Optimum**

cs.GT

11 pages, 8 figures

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.04787v2) [paper-pdf](http://arxiv.org/pdf/2410.04787v2)

**Authors**: Yuji Cao, Yue Chen

**Abstract**: This paper proposes a differentially private energy trading mechanism for prosumers in peer-to-peer (P2P) markets, offering provable privacy guarantees while approaching the Nash equilibrium with nearly socially optimal efficiency. We first model the P2P energy trading as a (generalized) Nash game and prove the vulnerability of traditional distributed algorithms to privacy attacks through an adversarial inference model. To address this challenge, we develop a privacy-preserving Nash equilibrium seeking algorithm incorporating carefully calibrated Laplacian noise. We prove that the proposed algorithm achieves $\epsilon$-differential privacy while converging in expectation to the Nash equilibrium with a suitable stepsize. Numerical experiments are conducted to evaluate the algorithm's robustness against privacy attacks, convergence behavior, and optimality compared to the non-private solution. Results demonstrate that our mechanism effectively protects prosumers' sensitive information while maintaining near-optimal market outcomes, offering a practical approach for privacy-preserving coordination in P2P markets.



## **44. Model Mimic Attack: Knowledge Distillation for Provably Transferable Adversarial Examples**

cs.LG

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15889v1) [paper-pdf](http://arxiv.org/pdf/2410.15889v1)

**Authors**: Kirill Lukyanov, Andrew Perminov, Denis Turdakov, Mikhail Pautov

**Abstract**: The vulnerability of artificial neural networks to adversarial perturbations in the black-box setting is widely studied in the literature. The majority of attack methods to construct these perturbations suffer from an impractically large number of queries required to find an adversarial example. In this work, we focus on knowledge distillation as an approach to conduct transfer-based black-box adversarial attacks and propose an iterative training of the surrogate model on an expanding dataset. This work is the first, to our knowledge, to provide provable guarantees on the success of knowledge distillation-based attack on classification neural networks: we prove that if the student model has enough learning capabilities, the attack on the teacher model is guaranteed to be found within the finite number of distillation iterations.



## **45. Vulnerabilities in Machine Learning-Based Voice Disorder Detection Systems**

cs.CR

7 pages, 17 figures, accepted for 16th IEEE INTERNATIONAL WORKSHOP ON  INFORMATION FORENSICS AND SECURITY (WIFS) 2024

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16341v1) [paper-pdf](http://arxiv.org/pdf/2410.16341v1)

**Authors**: Gianpaolo Perelli, Andrea Panzino, Roberto Casula, Marco Micheletto, Giulia Orrù, Gian Luca Marcialis

**Abstract**: The impact of voice disorders is becoming more widely acknowledged as a public health issue. Several machine learning-based classifiers with the potential to identify disorders have been used in recent studies to differentiate between normal and pathological voices and sounds. In this paper, we focus on analyzing the vulnerabilities of these systems by exploring the possibility of attacks that can reverse classification and compromise their reliability. Given the critical nature of personal health information, understanding which types of attacks are effective is a necessary first step toward improving the security of such systems. Starting from the original audios, we implement various attack methods, including adversarial, evasion, and pitching techniques, and evaluate how state-of-the-art disorder detection models respond to them. Our findings identify the most effective attack strategies, underscoring the need to address these vulnerabilities in machine-learning systems used in the healthcare domain.



## **46. NetSafe: Exploring the Topological Safety of Multi-agent Networks**

cs.MA

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15686v1) [paper-pdf](http://arxiv.org/pdf/2410.15686v1)

**Authors**: Miao Yu, Shilong Wang, Guibin Zhang, Junyuan Mao, Chenlong Yin, Qijiong Liu, Qingsong Wen, Kun Wang, Yang Wang

**Abstract**: Large language models (LLMs) have empowered nodes within multi-agent networks with intelligence, showing growing applications in both academia and industry. However, how to prevent these networks from generating malicious information remains unexplored with previous research on single LLM's safety be challenging to transfer. In this paper, we focus on the safety of multi-agent networks from a topological perspective, investigating which topological properties contribute to safer networks. To this end, we propose a general framework, NetSafe along with an iterative RelCom interaction to unify existing diverse LLM-based agent frameworks, laying the foundation for generalized topological safety research. We identify several critical phenomena when multi-agent networks are exposed to attacks involving misinformation, bias, and harmful information, termed as Agent Hallucination and Aggregation Safety. Furthermore, we find that highly connected networks are more susceptible to the spread of adversarial attacks, with task performance in a Star Graph Topology decreasing by 29.7%. Besides, our proposed static metrics aligned more closely with real-world dynamic evaluations than traditional graph-theoretic metrics, indicating that networks with greater average distances from attackers exhibit enhanced safety. In conclusion, our work introduces a new topological perspective on the safety of LLM-based multi-agent networks and discovers several unreported phenomena, paving the way for future research to explore the safety of such networks.



## **47. Patrol Security Game: Defending Against Adversary with Freedom in Attack Timing, Location, and Duration**

cs.AI

Under review of TCPS

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15600v1) [paper-pdf](http://arxiv.org/pdf/2410.15600v1)

**Authors**: Hao-Tsung Yang, Ting-Kai Weng, Ting-Yu Chang, Kin Sum Liu, Shan Lin, Jie Gao, Shih-Yu Tsai

**Abstract**: We explored the Patrol Security Game (PSG), a robotic patrolling problem modeled as an extensive-form Stackelberg game, where the attacker determines the timing, location, and duration of their attack. Our objective is to devise a patrolling schedule with an infinite time horizon that minimizes the attacker's payoff. We demonstrated that PSG can be transformed into a combinatorial minimax problem with a closed-form objective function. By constraining the defender's strategy to a time-homogeneous first-order Markov chain (i.e., the patroller's next move depends solely on their current location), we proved that the optimal solution in cases of zero penalty involves either minimizing the expected hitting time or return time, depending on the attacker model, and that these solutions can be computed efficiently. Additionally, we observed that increasing the randomness in the patrol schedule reduces the attacker's expected payoff in high-penalty cases. However, the minimax problem becomes non-convex in other scenarios. To address this, we formulated a bi-criteria optimization problem incorporating two objectives: expected maximum reward and entropy. We proposed three graph-based algorithms and one deep reinforcement learning model, designed to efficiently balance the trade-off between these two objectives. Notably, the third algorithm can identify the optimal deterministic patrol schedule, though its runtime grows exponentially with the number of patrol spots. Experimental results validate the effectiveness and scalability of our solutions, demonstrating that our approaches outperform state-of-the-art baselines on both synthetic and real-world crime datasets.



## **48. TrojanForge: Generating Adversarial Hardware Trojan Examples with Reinforcement Learning**

cs.CR

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2405.15184v2) [paper-pdf](http://arxiv.org/pdf/2405.15184v2)

**Authors**: Amin Sarihi, Peter Jamieson, Ahmad Patooghy, Abdel-Hameed A. Badawy

**Abstract**: The Hardware Trojan (HT) problem can be thought of as a continuous game between attackers and defenders, each striving to outsmart the other by leveraging any available means for an advantage. Machine Learning (ML) has recently played a key role in advancing HT research. Various novel techniques, such as Reinforcement Learning (RL) and Graph Neural Networks (GNNs), have shown HT insertion and detection capabilities. HT insertion with ML techniques, specifically, has seen a spike in research activity due to the shortcomings of conventional HT benchmarks and the inherent human design bias that occurs when we create them. This work continues this innovation by presenting a tool called TrojanForge, capable of generating HT adversarial examples that defeat HT detectors; demonstrating the capabilities of GAN-like adversarial tools for automatic HT insertion. We introduce an RL environment where the RL insertion agent interacts with HT detectors in an insertion-detection loop where the agent collects rewards based on its success in bypassing HT detectors. Our results show that this process helps inserted HTs evade various HT detectors, achieving high attack success percentages. This tool provides insight into why HT insertion fails in some instances and how we can leverage this knowledge in defense.



## **49. BRC20 Pinning Attack**

cs.CR

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2410.11295v2) [paper-pdf](http://arxiv.org/pdf/2410.11295v2)

**Authors**: Minfeng Qi, Qin Wang, Zhipeng Wang, Lin Zhong, Tianqing Zhu, Shiping Chen, William Knottenbelt

**Abstract**: BRC20 tokens are a type of non-fungible asset on the Bitcoin network. They allow users to embed customized content within Bitcoin satoshis. The related token frenzy has reached a market size of US$2,650b over the past year (2023Q3-2024Q3). However, this intuitive design has not undergone serious security scrutiny.   We present the first in-depth analysis of the BRC20 transfer mechanism and identify a critical attack vector. A typical BRC20 transfer involves two bundled on-chain transactions with different fee levels: the first (i.e., Tx1) with a lower fee inscribes the transfer request, while the second (i.e., Tx2) with a higher fee finalizes the actual transfer. We find that an adversary can exploit this by sending a manipulated fee transaction (falling between the two fee levels), which allows Tx1 to be processed while Tx2 remains pinned in the mempool. This locks the BRC20 liquidity and disrupts normal transfers for users. We term this BRC20 pinning attack.   Our attack exposes an inherent design flaw that can be applied to 90+% inscription-based tokens within the Bitcoin ecosystem.   We also conducted the attack on Binance's ORDI hot wallet (the most prevalent BRC20 token and the most active wallet), resulting in a temporary suspension of ORDI withdrawals on Binance for 3.5 hours, which were shortly resumed after our communication.



## **50. Revisit, Extend, and Enhance Hessian-Free Influence Functions**

cs.LG

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2405.17490v2) [paper-pdf](http://arxiv.org/pdf/2405.17490v2)

**Authors**: Ziao Yang, Han Yue, Jian Chen, Hongfu Liu

**Abstract**: Influence functions serve as crucial tools for assessing sample influence in model interpretation, subset training set selection, noisy label detection, and more. By employing the first-order Taylor extension, influence functions can estimate sample influence without the need for expensive model retraining. However, applying influence functions directly to deep models presents challenges, primarily due to the non-convex nature of the loss function and the large size of model parameters. This difficulty not only makes computing the inverse of the Hessian matrix costly but also renders it non-existent in some cases. Various approaches, including matrix decomposition, have been explored to expedite and approximate the inversion of the Hessian matrix, with the aim of making influence functions applicable to deep models. In this paper, we revisit a specific, albeit naive, yet effective approximation method known as TracIn. This method substitutes the inverse of the Hessian matrix with an identity matrix. We provide deeper insights into why this simple approximation method performs well. Furthermore, we extend its applications beyond measuring model utility to include considerations of fairness and robustness. Finally, we enhance TracIn through an ensemble strategy. To validate its effectiveness, we conduct experiments on synthetic data and extensive evaluations on noisy label detection, sample selection for large language model fine-tuning, and defense against adversarial attacks.



