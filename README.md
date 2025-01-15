# Latest Adversarial Attack Papers
**update at 2025-01-15 23:32:23**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Exploring Robustness of LLMs to Sociodemographically-Conditioned Paraphrasing**

cs.CL

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08276v1) [paper-pdf](http://arxiv.org/pdf/2501.08276v1)

**Authors**: Pulkit Arora, Akbar Karimi, Lucie Flek

**Abstract**: Large Language Models (LLMs) have shown impressive performance in various NLP tasks. However, there are concerns about their reliability in different domains of linguistic variations. Many works have proposed robustness evaluation measures for local adversarial attacks, but we need globally robust models unbiased to different language styles. We take a broader approach to explore a wider range of variations across sociodemographic dimensions to perform structured reliability tests on the reasoning capacity of language models. We extend the SocialIQA dataset to create diverse paraphrased sets conditioned on sociodemographic styles. The assessment aims to provide a deeper understanding of LLMs in (a) their capability of generating demographic paraphrases with engineered prompts and (b) their reasoning capabilities in real-world, complex language scenarios. We also explore measures such as perplexity, explainability, and ATOMIC performance of paraphrases for fine-grained reliability analysis of LLMs on these sets. We find that demographic-specific paraphrasing significantly impacts the performance of language models, indicating that the subtleties of language variations remain a significant challenge. The code and dataset will be made available for reproducibility and future research.



## **2. Towards an End-to-End (E2E) Adversarial Learning and Application in the Physical World**

cs.CV

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08258v1) [paper-pdf](http://arxiv.org/pdf/2501.08258v1)

**Authors**: Dudi Biton, Jacob Shams, Koda Satoru, Asaf Shabtai, Yuval Elovici, Ben Nassi

**Abstract**: The traditional learning process of patch-based adversarial attacks, conducted in the digital domain and then applied in the physical domain (e.g., via printed stickers), may suffer from reduced performance due to adversarial patches' limited transferability from the digital domain to the physical domain. Given that previous studies have considered using projectors to apply adversarial attacks, we raise the following question: can adversarial learning (i.e., patch generation) be performed entirely in the physical domain with a projector? In this work, we propose the Physical-domain Adversarial Patch Learning Augmentation (PAPLA) framework, a novel end-to-end (E2E) framework that converts adversarial learning from the digital domain to the physical domain using a projector. We evaluate PAPLA across multiple scenarios, including controlled laboratory settings and realistic outdoor environments, demonstrating its ability to ensure attack success compared to conventional digital learning-physical application (DL-PA) methods. We also analyze the impact of environmental factors, such as projection surface color, projector strength, ambient light, distance, and angle of the target object relative to the camera, on the effectiveness of projected patches. Finally, we demonstrate the feasibility of the attack against a parked car and a stop sign in a real-world outdoor environment. Our results show that under specific conditions, E2E adversarial learning in the physical domain eliminates the transferability issue and ensures evasion by object detectors. Finally, we provide insights into the challenges and opportunities of applying adversarial learning in the physical domain and explain where such an approach is more effective than using a sticker.



## **3. SoK: Design, Vulnerabilities, and Security Measures of Cryptocurrency Wallets**

cs.CR

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2307.12874v4) [paper-pdf](http://arxiv.org/pdf/2307.12874v4)

**Authors**: Yimika Erinle, Yathin Kethepalli, Yebo Feng, Jiahua Xu

**Abstract**: With the advent of decentralised digital currencies powered by blockchain technology, a new era of peer-to-peer transactions has commenced. The rapid growth of the cryptocurrency economy has led to increased use of transaction-enabling wallets, making them a focal point for security risks. As the frequency of wallet-related incidents rises, there is a critical need for a systematic approach to measure and evaluate these attacks, drawing lessons from past incidents to enhance wallet security. In response, we introduce a multi-dimensional design taxonomy for existing and novel wallets with various design decisions. We classify existing industry wallets based on this taxonomy, identify previously occurring vulnerabilities and discuss the security implications of design decisions. We also systematise threats to the wallet mechanism and analyse the adversary's goals, capabilities and required knowledge. We present a multi-layered attack framework and investigate 84 incidents between 2012 and 2024, accounting for $5.4B. Following this, we classify defence implementations for these attacks on the precautionary and remedial axes. We map the mechanism and design decisions to vulnerabilities, attacks, and possible defence methods to discuss various insights.



## **4. I Can Find You in Seconds! Leveraging Large Language Models for Code Authorship Attribution**

cs.SE

12 pages, 5 figures,

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08165v1) [paper-pdf](http://arxiv.org/pdf/2501.08165v1)

**Authors**: Soohyeon Choi, Yong Kiam Tan, Mark Huasong Meng, Mohamed Ragab, Soumik Mondal, David Mohaisen, Khin Mi Mi Aung

**Abstract**: Source code authorship attribution is important in software forensics, plagiarism detection, and protecting software patch integrity. Existing techniques often rely on supervised machine learning, which struggles with generalization across different programming languages and coding styles due to the need for large labeled datasets. Inspired by recent advances in natural language authorship analysis using large language models (LLMs), which have shown exceptional performance without task-specific tuning, this paper explores the use of LLMs for source code authorship attribution.   We present a comprehensive study demonstrating that state-of-the-art LLMs can successfully attribute source code authorship across different languages. LLMs can determine whether two code snippets are written by the same author with zero-shot prompting, achieving a Matthews Correlation Coefficient (MCC) of 0.78, and can attribute code authorship from a small set of reference code snippets via few-shot learning, achieving MCC of 0.77. Additionally, LLMs show some adversarial robustness against misattribution attacks.   Despite these capabilities, we found that naive prompting of LLMs does not scale well with a large number of authors due to input token limitations. To address this, we propose a tournament-style approach for large-scale attribution. Evaluating this approach on datasets of C++ (500 authors, 26,355 samples) and Java (686 authors, 55,267 samples) code from GitHub, we achieve classification accuracy of up to 65% for C++ and 68.7% for Java using only one reference per author. These results open new possibilities for applying LLMs to code authorship attribution in cybersecurity and software engineering.



## **5. Set-Based Training for Neural Network Verification**

cs.LG

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2401.14961v3) [paper-pdf](http://arxiv.org/pdf/2401.14961v3)

**Authors**: Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can significantly affect the outputs of a neural network. Therefore, to ensure safety of safety-critical environments, the robustness of a neural network must be formally verified against input perturbations, e.g., from noisy sensors. To improve the robustness of neural networks and thus simplify the formal verification, we present a novel set-based training procedure in which we compute the set of possible outputs given the set of possible inputs and compute for the first time a gradient set, i.e., each possible output has a different gradient. Therefore, we can directly reduce the size of the output enclosure by choosing gradients toward its center. Small output enclosures increase the robustness of a neural network and, at the same time, simplify its formal verification. The latter benefit is due to the fact that a larger size of propagated sets increases the conservatism of most verification methods. Our extensive evaluation demonstrates that set-based training produces robust neural networks with competitive performance, which can be verified using fast (polynomial-time) verification algorithms due to the reduced output set.



## **6. Gandalf the Red: Adaptive Security for LLMs**

cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.07927v1) [paper-pdf](http://arxiv.org/pdf/2501.07927v1)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Natalie Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and rigorously expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack datasets. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications. Code is available at \href{https://github.com/lakeraai/dsec-gandalf}{\texttt{https://github.com/lakeraai/dsec-gandalf}}.



## **7. VENOM: Text-driven Unrestricted Adversarial Example Generation with Diffusion Models**

cs.CV

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.07922v1) [paper-pdf](http://arxiv.org/pdf/2501.07922v1)

**Authors**: Hui Kuurila-Zhang, Haoyu Chen, Guoying Zhao

**Abstract**: Adversarial attacks have proven effective in deceiving machine learning models by subtly altering input images, motivating extensive research in recent years. Traditional methods constrain perturbations within $l_p$-norm bounds, but advancements in Unrestricted Adversarial Examples (UAEs) allow for more complex, generative-model-based manipulations. Diffusion models now lead UAE generation due to superior stability and image quality over GANs. However, existing diffusion-based UAE methods are limited to using reference images and face challenges in generating Natural Adversarial Examples (NAEs) directly from random noise, often producing uncontrolled or distorted outputs. In this work, we introduce VENOM, the first text-driven framework for high-quality unrestricted adversarial examples generation through diffusion models. VENOM unifies image content generation and adversarial synthesis into a single reverse diffusion process, enabling high-fidelity adversarial examples without sacrificing attack success rate (ASR). To stabilize this process, we incorporate an adaptive adversarial guidance strategy with momentum, ensuring that the generated adversarial examples $x^*$ align with the distribution $p(x)$ of natural images. Extensive experiments demonstrate that VENOM achieves superior ASR and image quality compared to prior methods, marking a significant advancement in adversarial example generation and providing insights into model vulnerabilities for improved defense development.



## **8. Can Go AIs be adversarially robust?**

cs.LG

63 pages, AAAI 2025

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2406.12843v3) [paper-pdf](http://arxiv.org/pdf/2406.12843v3)

**Authors**: Tom Tseng, Euan McLean, Kellin Pelrine, Tony T. Wang, Adam Gleave

**Abstract**: Prior work found that superhuman Go AIs can be defeated by simple adversarial strategies, especially "cyclic" attacks. In this paper, we study whether adding natural countermeasures can achieve robustness in Go, a favorable domain for robustness since it benefits from incredible average-case capability and a narrow, innately adversarial setting. We test three defenses: adversarial training on hand-constructed positions, iterated adversarial training, and changing the network architecture. We find that though some of these defenses protect against previously discovered attacks, none withstand freshly trained adversaries. Furthermore, most of the reliably effective attacks these adversaries discover are different realizations of the same overall class of cyclic attacks. Our results suggest that building robust AI systems is challenging even with extremely superhuman systems in some of the most tractable settings, and highlight two key gaps: efficient generalization of defenses, and diversity in training. For interactive examples of attacks and a link to our codebase, see https://goattack.far.ai.



## **9. Don't Command, Cultivate: An Exploratory Study of System-2 Alignment**

cs.CL

In this version, the DPO and reinforcement learning methods have been  added

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2411.17075v5) [paper-pdf](http://arxiv.org/pdf/2411.17075v5)

**Authors**: Yuhang Wang, Yuxiang Zhang, Yanxu Zhu, Xinyan Wen, Jitao Sang

**Abstract**: The o1 system card identifies the o1 models as the most robust within OpenAI, with their defining characteristic being the progression from rapid, intuitive thinking to slower, more deliberate reasoning. This observation motivated us to investigate the influence of System-2 thinking patterns on model safety. In our preliminary research, we conducted safety evaluations of the o1 model, including complex jailbreak attack scenarios using adversarial natural language prompts and mathematical encoding prompts. Our findings indicate that the o1 model demonstrates relatively improved safety performance; however, it still exhibits vulnerabilities, particularly against jailbreak attacks employing mathematical encoding. Through detailed case analysis, we identified specific patterns in the o1 model's responses. We also explored the alignment of System-2 safety in open-source models using prompt engineering and supervised fine-tuning techniques. Experimental results show that some simple methods to encourage the model to carefully scrutinize user requests are beneficial for model safety. Additionally, we proposed a implementation plan for process supervision to enhance safety alignment. The implementation details and experimental results will be provided in future versions.



## **10. A Survey of Early Exit Deep Neural Networks in NLP**

cs.LG

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07670v1) [paper-pdf](http://arxiv.org/pdf/2501.07670v1)

**Authors**: Divya Jyoti Bajpai, Manjesh Kumar Hanawal

**Abstract**: Deep Neural Networks (DNNs) have grown increasingly large in size to achieve state of the art performance across a wide range of tasks. However, their high computational requirements make them less suitable for resource-constrained applications. Also, real-world datasets often consist of a mixture of easy and complex samples, necessitating adaptive inference mechanisms that account for sample difficulty. Early exit strategies offer a promising solution by enabling adaptive inference, where simpler samples are classified using the initial layers of the DNN, thereby accelerating the overall inference process. By attaching classifiers at different layers, early exit methods not only reduce inference latency but also improve the model robustness against adversarial attacks. This paper presents a comprehensive survey of early exit methods and their applications in NLP.



## **11. SecAlign: Defending Against Prompt Injection with Preference Optimization**

cs.CR

Key words: prompt injection defense, LLM security, LLM-integrated  applications

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2410.05451v2) [paper-pdf](http://arxiv.org/pdf/2410.05451v2)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction.   To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to around 0%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, our defended models are still practical with similar utility to the one before our defensive training. Our code is at https://github.com/facebookresearch/SecAlign



## **12. Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards**

cs.LG

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07493v1) [paper-pdf](http://arxiv.org/pdf/2501.07493v1)

**Authors**: Yangsibo Huang, Milad Nasr, Anastasios Angelopoulos, Nicholas Carlini, Wei-Lin Chiang, Christopher A. Choquette-Choo, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Ken Ziyu Liu, Ion Stoica, Florian Tramer, Chiyuan Zhang

**Abstract**: It is now common to evaluate Large Language Models (LLMs) by having humans manually vote to evaluate model outputs, in contrast to typical benchmarks that evaluate knowledge or skill at some particular task. Chatbot Arena, the most popular benchmark of this type, ranks models by asking users to select the better response between two randomly selected models (without revealing which model was responsible for the generations). These platforms are widely trusted as a fair and accurate measure of LLM capabilities. In this paper, we show that if bot protection and other defenses are not implemented, these voting-based benchmarks are potentially vulnerable to adversarial manipulation. Specifically, we show that an attacker can alter the leaderboard (to promote their favorite model or demote competitors) at the cost of roughly a thousand votes (verified in a simulated, offline version of Chatbot Arena). Our attack consists of two steps: first, we show how an attacker can determine which model was used to generate a given reply with more than $95\%$ accuracy; and then, the attacker can use this information to consistently vote for (or against) a target model. Working with the Chatbot Arena developers, we identify, propose, and implement mitigations to improve the robustness of Chatbot Arena against adversarial manipulation, which, based on our analysis, substantially increases the cost of such attacks. Some of these defenses were present before our collaboration, such as bot protection with Cloudflare, malicious user detection, and rate limiting. Others, including reCAPTCHA and login are being integrated to strengthen the security in Chatbot Arena.



## **13. Agentic Copyright Watermarking against Adversarial Evidence Forgery with Purification-Agnostic Curriculum Proxy Learning**

cs.CV

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2409.01541v2) [paper-pdf](http://arxiv.org/pdf/2409.01541v2)

**Authors**: Erjin Bao, Ching-Chun Chang, Hanrui Wang, Isao Echizen

**Abstract**: With the proliferation of AI agents in various domains, protecting the ownership of AI models has become crucial due to the significant investment in their development. Unauthorized use and illegal distribution of these models pose serious threats to intellectual property, necessitating effective copyright protection measures. Model watermarking has emerged as a key technique to address this issue, embedding ownership information within models to assert rightful ownership during copyright disputes. This paper presents several contributions to model watermarking: a self-authenticating black-box watermarking protocol using hash techniques, a study on evidence forgery attacks using adversarial perturbations, a proposed defense involving a purification step to counter adversarial attacks, and a purification-agnostic curriculum proxy learning method to enhance watermark robustness and model performance. Experimental results demonstrate the effectiveness of these approaches in improving the security, reliability, and performance of watermarked models.



## **14. MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework**

cs.LG

Under Review of CVPR 2025

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07251v1) [paper-pdf](http://arxiv.org/pdf/2501.07251v1)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Fei Liu, Zhichao Lu, Qingfu Zhang, Zhenkun Wang

**Abstract**: Crafting adversarial examples is crucial for evaluating and enhancing the robustness of Deep Neural Networks (DNNs), presenting a challenge equivalent to maximizing a non-differentiable 0-1 loss function.   However, existing single objective methods, namely adversarial attacks focus on a surrogate loss function, do not fully harness the benefits of engaging multiple loss functions, as a result of insufficient understanding of their synergistic and conflicting nature.   To overcome these limitations, we propose the Multi-Objective Set-based Attack (MOS Attack), a novel adversarial attack framework leveraging multiple loss functions and automatically uncovering their interrelations.   The MOS Attack adopts a set-based multi-objective optimization strategy, enabling the incorporation of numerous loss functions without additional parameters.   It also automatically mines synergistic patterns among various losses, facilitating the generation of potent adversarial attacks with fewer objectives.   Extensive experiments have shown that our MOS Attack outperforms single-objective attacks. Furthermore, by harnessing the identified synergistic patterns, MOS Attack continues to show superior results with a reduced number of loss functions.



## **15. An Enhanced Zeroth-Order Stochastic Frank-Wolfe Framework for Constrained Finite-Sum Optimization**

cs.LG

35 pages, 4 figures, 3 tables

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07201v1) [paper-pdf](http://arxiv.org/pdf/2501.07201v1)

**Authors**: Haishan Ye, Yinghui Huang, Hao Di, Xiangyu Chang

**Abstract**: We propose an enhanced zeroth-order stochastic Frank-Wolfe framework to address constrained finite-sum optimization problems, a structure prevalent in large-scale machine-learning applications. Our method introduces a novel double variance reduction framework that effectively reduces the gradient approximation variance induced by zeroth-order oracles and the stochastic sampling variance from finite-sum objectives. By leveraging this framework, our algorithm achieves significant improvements in query efficiency, making it particularly well-suited for high-dimensional optimization tasks. Specifically, for convex objectives, the algorithm achieves a query complexity of O(d \sqrt{n}/\epsilon ) to find an epsilon-suboptimal solution, where d is the dimensionality and n is the number of functions in the finite-sum objective. For non-convex objectives, it achieves a query complexity of O(d^{3/2}\sqrt{n}/\epsilon^2 ) without requiring the computation ofd partial derivatives at each iteration. These complexities are the best known among zeroth-order stochastic Frank-Wolfe algorithms that avoid explicit gradient calculations. Empirical experiments on convex and non-convex machine learning tasks, including sparse logistic regression, robust classification, and adversarial attacks on deep networks, validate the computational efficiency and scalability of our approach. Our algorithm demonstrates superior performance in both convergence rate and query complexity compared to existing methods.



## **16. Bitcoin Under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

cs.CR

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2411.11702v2) [paper-pdf](http://arxiv.org/pdf/2411.11702v2)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: The security of Bitcoin protocols is deeply dependent on the incentives provided to miners, which come from a combination of block rewards and transaction fees. As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, leads to the emergence of new security threats or intensifies existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   This paper presents a reinforcement learning-based tool to develop mining strategies under a more realistic volatile model. We employ the Asynchronous Advantage Actor-Critic (A3C) algorithm, which efficiently handles dynamic environments, such as the Bitcoin mempool, to derive near-optimal mining strategies when interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   We revisit the Bitcoin security threshold presented in the WeRLman paper and demonstrate that the implicit predictability of valuable transaction arrivals in this model leads to an underestimation of the reported threshold. Additionally, we show that, while adversarial strategies like selfish mining under the fixed reward model incur an initial loss period of at least two weeks, the transition toward a transaction-fee era incentivizes mining pools to abandon honest mining for immediate profits. This incentive is expected to become more significant as the protocol reward approaches zero in the future.



## **17. Protego: Detecting Adversarial Examples for Vision Transformers via Intrinsic Capabilities**

cs.CV

Accepted by IEEE MetaCom 2024

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07044v1) [paper-pdf](http://arxiv.org/pdf/2501.07044v1)

**Authors**: Jialin Wu, Kaikai Pan, Yanjiao Chen, Jiangyi Deng, Shengyuan Pang, Wenyuan Xu

**Abstract**: Transformer models have excelled in natural language tasks, prompting the vision community to explore their implementation in computer vision problems. However, these models are still influenced by adversarial examples. In this paper, we investigate the attack capabilities of six common adversarial attacks on three pretrained ViT models to reveal the vulnerability of ViT models. To understand and analyse the bias in neural network decisions when the input is adversarial, we use two visualisation techniques that are attention rollout and grad attention rollout. To prevent ViT models from adversarial attack, we propose Protego, a detection framework that leverages the transformer intrinsic capabilities to detection adversarial examples of ViT models. Nonetheless, this is challenging due to a diversity of attack strategies that may be adopted by adversaries. Inspired by the attention mechanism, we know that the token of prediction contains all the information from the input sample. Additionally, the attention region for adversarial examples differs from that of normal examples. Given these points, we can train a detector that achieves superior performance than existing detection methods to identify adversarial examples. Our experiments have demonstrated the high effectiveness of our detection method. For these six adversarial attack methods, our detector's AUC scores all exceed 0.95. Protego may advance investigations in metaverse security.



## **18. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

cs.MM

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2308.12636v4) [paper-pdf](http://arxiv.org/pdf/2308.12636v4)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Hanwang Zhang, Hang Su, Richang Hong

**Abstract**: The integration of visual and textual data in Vision-Language Pre-training (VLP) models is crucial for enhancing vision-language understanding. However, the adversarial robustness of these models, especially in the alignment of image-text features, has not yet been sufficiently explored. In this paper, we introduce a novel gradient-based multimodal adversarial attack method, underpinned by contrastive learning, to improve the transferability of multimodal adversarial samples in VLP models. This method concurrently generates adversarial texts and images within imperceptive perturbation, employing both image-text and intra-modal contrastive loss. We evaluate the effectiveness of our approach on image-text retrieval and visual entailment tasks, using publicly available datasets in a black-box setting. Extensive experiments indicate a significant advancement over existing single-modal transfer-based adversarial attack methods and current multimodal adversarial attack approaches.



## **19. Mitigating Low-Frequency Bias: Feature Recalibration and Frequency Attention Regularization for Adversarial Robustness**

cs.CV

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2407.04016v2) [paper-pdf](http://arxiv.org/pdf/2407.04016v2)

**Authors**: Kejia Zhang, Juanjuan Weng, Yuanzheng Cai, Zhiming Luo, Shaozi Li

**Abstract**: Ensuring the robustness of deep neural networks against adversarial attacks remains a fundamental challenge in computer vision. While adversarial training (AT) has emerged as a promising defense strategy, our analysis reveals a critical limitation: AT-trained models exhibit a bias toward low-frequency features while neglecting high-frequency components. This bias is particularly concerning as each frequency component carries distinct and crucial information: low-frequency features encode fundamental structural patterns, while high-frequency features capture intricate details and textures. To address this limitation, we propose High-Frequency Feature Disentanglement and Recalibration (HFDR), a novel module that strategically separates and recalibrates frequency-specific features to capture latent semantic cues. We further introduce frequency attention regularization to harmonize feature extraction across the frequency spectrum and mitigate the inherent low-frequency bias of AT. Extensive experiments demonstrate our method's superior performance against white-box attacks and transfer attacks, while exhibiting strong generalization capabilities across diverse scenarios.



## **20. ModelShield: Adaptive and Robust Watermark against Model Extraction Attack**

cs.CR

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2405.02365v4) [paper-pdf](http://arxiv.org/pdf/2405.02365v4)

**Authors**: Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai, Minghu Jiang, Yongfeng Huang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.



## **21. ZOQO: Zero-Order Quantized Optimization**

cs.LG

Accepted to ICASSP 2025

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.06736v1) [paper-pdf](http://arxiv.org/pdf/2501.06736v1)

**Authors**: Noga Bar, Raja Giryes

**Abstract**: The increasing computational and memory demands in deep learning present significant challenges, especially in resource-constrained environments. We introduce a zero-order quantized optimization (ZOQO) method designed for training models with quantized parameters and operations. Our approach leverages zero-order approximations of the gradient sign and adapts the learning process to maintain the parameters' quantization without the need for full-precision gradient calculations. We demonstrate the effectiveness of ZOQO through experiments in fine-tuning of large language models and black-box adversarial attacks. Despite the limitations of zero-order and quantized operations training, our method achieves competitive performance compared to full-precision methods, highlighting its potential for low-resource environments.



## **22. Measuring the Robustness of Reference-Free Dialogue Evaluation Systems**

cs.CL

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.06728v1) [paper-pdf](http://arxiv.org/pdf/2501.06728v1)

**Authors**: Justin Vasselli, Adam Nohejl, Taro Watanabe

**Abstract**: Advancements in dialogue systems powered by large language models (LLMs) have outpaced the development of reliable evaluation metrics, particularly for diverse and creative responses. We present a benchmark for evaluating the robustness of reference-free dialogue metrics against four categories of adversarial attacks: speaker tag prefixes, static responses, ungrammatical responses, and repeated conversational context. We analyze metrics such as DialogRPT, UniEval, and PromptEval -- a prompt-based method leveraging LLMs -- across grounded and ungrounded datasets. By examining both their correlation with human judgment and susceptibility to adversarial attacks, we find that these two axes are not always aligned; metrics that appear to be equivalent when judged by traditional benchmarks may, in fact, vary in their scores of adversarial responses. These findings motivate the development of nuanced evaluation frameworks to address real-world dialogue challenges.



## **23. Towards Adversarially Robust Deep Metric Learning**

cs.LG

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.01025v2) [paper-pdf](http://arxiv.org/pdf/2501.01025v2)

**Authors**: Xiaopeng Ke

**Abstract**: Deep Metric Learning (DML) has shown remarkable successes in many domains by taking advantage of powerful deep neural networks. Deep neural networks are prone to adversarial attacks and could be easily fooled by adversarial examples. The current progress on this robustness issue is mainly about deep classification models but pays little attention to DML models. Existing works fail to thoroughly inspect the robustness of DML and neglect an important DML scenario, the clustering-based inference. In this work, we first point out the robustness issue of DML models in clustering-based inference scenarios. We find that, for the clustering-based inference, existing defenses designed DML are unable to be reused and the adaptions of defenses designed for deep classification models cannot achieve satisfactory robustness performance. To alleviate the hazard of adversarial examples, we propose a new defense, the Ensemble Adversarial Training (EAT), which exploits ensemble learning and adversarial training. EAT promotes the diversity of the ensemble, encouraging each model in the ensemble to have different robustness features, and employs a self-transferring mechanism to make full use of the robustness statistics of the whole ensemble in the update of every single model. We evaluate the EAT method on three widely-used datasets with two popular model architectures. The results show that the proposed EAT method greatly outperforms the adaptions of defenses designed for deep classification models.



## **24. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-01-11    [abs](http://arxiv.org/abs/2410.17910v2) [paper-pdf](http://arxiv.org/pdf/2410.17910v2)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.



## **25. LayerMix: Enhanced Data Augmentation through Fractal Integration for Robust Deep Learning**

cs.CV

**SubmitDate**: 2025-01-11    [abs](http://arxiv.org/abs/2501.04861v2) [paper-pdf](http://arxiv.org/pdf/2501.04861v2)

**Authors**: Hafiz Mughees Ahmad, Dario Morle, Afshin Rahimi

**Abstract**: Deep learning models have demonstrated remarkable performance across various computer vision tasks, yet their vulnerability to distribution shifts remains a critical challenge. Despite sophisticated neural network architectures, existing models often struggle to maintain consistent performance when confronted with Out-of-Distribution (OOD) samples, including natural corruptions, adversarial perturbations, and anomalous patterns. We introduce LayerMix, an innovative data augmentation approach that systematically enhances model robustness through structured fractal-based image synthesis. By meticulously integrating structural complexity into training datasets, our method generates semantically consistent synthetic samples that significantly improve neural network generalization capabilities. Unlike traditional augmentation techniques that rely on random transformations, LayerMix employs a structured mixing pipeline that preserves original image semantics while introducing controlled variability. Extensive experiments across multiple benchmark datasets, including CIFAR-10, CIFAR-100, ImageNet-200, and ImageNet-1K demonstrate LayerMixs superior performance in classification accuracy and substantially enhances critical Machine Learning (ML) safety metrics, including resilience to natural image corruptions, robustness against adversarial attacks, improved model calibration and enhanced prediction consistency. LayerMix represents a significant advancement toward developing more reliable and adaptable artificial intelligence systems by addressing the fundamental challenges of deep learning generalization. The code is available at https://github.com/ahmadmughees/layermix.



## **26. Effective Backdoor Mitigation in Vision-Language Models Depends on the Pre-training Objective**

cs.LG

Accepted at TMLR (https://openreview.net/forum?id=Conma3qnaT)

**SubmitDate**: 2025-01-11    [abs](http://arxiv.org/abs/2311.14948v4) [paper-pdf](http://arxiv.org/pdf/2311.14948v4)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Pin-Yu Chen, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in multimodal models, such as CleanCLIP, which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives that lead to higher zero-shot classification performance correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP, even with extensive hyperparameter tuning, is ineffective in poison removal when stronger pre-training objectives are used. Our findings underscore critical considerations for ML practitioners who train models using large-scale web-curated data and are concerned about potential backdoor threats.



## **27. Privacy-Preserving Distributed Defense Framework for DC Microgrids Against Exponentially Unbounded False Data Injection Attacks**

eess.SY

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.00588v2) [paper-pdf](http://arxiv.org/pdf/2501.00588v2)

**Authors**: Yi Zhang, Mohamadamin Rajabinezhad, Yichao Wang, Junbo Zhao, Shan Zuo

**Abstract**: This paper introduces a novel, fully distributed control framework for DC microgrids, enhancing resilience against exponentially unbounded false data injection (EU-FDI) attacks. Our framework features a consensus-based secondary control for each converter, effectively addressing these advanced threats. To further safeguard sensitive operational data, a privacy-preserving mechanism is incorporated into the control design, ensuring that critical information remains secure even under adversarial conditions. Rigorous Lyapunov stability analysis confirms the framework's ability to maintain critical DC microgrid operations like voltage regulation and load sharing under EU-FDI threats. The framework's practicality is validated through hardware-in-the-loop experiments, demonstrating its enhanced resilience and robust privacy protection against the complex challenges posed by quick variant FDI attacks.



## **28. Pixel Is Not A Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models**

cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.11810v2) [paper-pdf](http://arxiv.org/pdf/2408.11810v2)

**Authors**: Chun-Yen Shih, Li-Xuan Peng, Jia-Wei Liao, Ernie Chu, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Diffusion Models have emerged as powerful generative models for high-quality image synthesis, with many subsequent image editing techniques based on them. However, the ease of text-based image editing introduces significant risks, such as malicious editing for scams or intellectual property infringement. Previous works have attempted to safeguard images from diffusion-based editing by adding imperceptible perturbations. These methods are costly and specifically target prevalent Latent Diffusion Models (LDMs), while Pixel-domain Diffusion Models (PDMs) remain largely unexplored and robust against such attacks. Our work addresses this gap by proposing a novel attack framework, AtkPDM. AtkPDM is mainly composed of a feature representation attacking loss that exploits vulnerabilities in denoising UNets and a latent optimization strategy to enhance the naturalness of adversarial images. Extensive experiments demonstrate the effectiveness of our approach in attacking dominant PDM-based editing methods (e.g., SDEdit) while maintaining reasonable fidelity and robustness against common defense methods. Additionally, our framework is extensible to LDMs, achieving comparable performance to existing approaches.



## **29. Adversarial Detection by Approximation of Ensemble Boundary**

cs.LG

27 pages, 7 figures, 5 tables

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2211.10227v5) [paper-pdf](http://arxiv.org/pdf/2211.10227v5)

**Authors**: T. Windeatt

**Abstract**: Despite being effective in many application areas, Deep Neural Networks (DNNs) are vulnerable to being attacked. In object recognition, the attack takes the form of a small perturbation added to an image, that causes the DNN to misclassify, but to a human appears no different. Adversarial attacks lead to defences that are themselves subject to attack, and the attack/ defence strategies provide important information about the properties of DNNs. In this paper, a novel method of detecting adversarial attacks is proposed for an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The ensemble is combined using Walsh coefficients which are capable of approximating Boolean functions and thereby controlling the decision boundary complexity. The hypothesis in this paper is that decision boundaries with high curvature allow adversarial perturbations to be found, but change the curvature of the decision boundary, which is then approximated in a different way by Walsh coefficients compared to the clean images. Besides controlling boundary complexity, the coefficients also measure the correlation with class labels, which may aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area.



## **30. Beyond Optimal Fault Tolerance**

cs.DC

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.06044v1) [paper-pdf](http://arxiv.org/pdf/2501.06044v1)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal ``recoverable fault-tolerance'' achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic ``recovery procedure'' that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.



## **31. Effective faking of verbal deception detection with target-aligned adversarial attacks**

cs.CL

preprint

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05962v1) [paper-pdf](http://arxiv.org/pdf/2501.05962v1)

**Authors**: Bennett Kleinberg, Riccardo Loconte, Bruno Verschuere

**Abstract**: Background: Deception detection through analysing language is a promising avenue using both human judgments and automated machine learning judgments. For both forms of credibility assessment, automated adversarial attacks that rewrite deceptive statements to appear truthful pose a serious threat. Methods: We used a dataset of 243 truthful and 262 fabricated autobiographical stories in a deception detection task for humans and machine learning models. A large language model was tasked to rewrite deceptive statements so that they appear truthful. In Study 1, humans who made a deception judgment or used the detailedness heuristic and two machine learning models (a fine-tuned language model and a simple n-gram model) judged original or adversarial modifications of deceptive statements. In Study 2, we manipulated the target alignment of the modifications, i.e. tailoring the attack to whether the statements would be assessed by humans or computer models. Results: When adversarial modifications were aligned with their target, human (d=-0.07 and d=-0.04) and machine judgments (51% accuracy) dropped to the chance level. When the attack was not aligned with the target, both human heuristics judgments (d=0.30 and d=0.36) and machine learning predictions (63-78%) were significantly better than chance. Conclusions: Easily accessible language models can effectively help anyone fake deception detection efforts both by humans and machine learning models. Robustness against adversarial modifications for humans and machines depends on that target alignment. We close with suggestions on advancing deception research with adversarial attack designs.



## **32. Towards Backdoor Stealthiness in Model Parameter Space**

cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05928v1) [paper-pdf](http://arxiv.org/pdf/2501.05928v1)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Stjepan Picek

**Abstract**: Recent research on backdoor stealthiness focuses mainly on indistinguishable triggers in input space and inseparable backdoor representations in feature space, aiming to circumvent backdoor defenses that examine these respective spaces. However, existing backdoor attacks are typically designed to resist a specific type of backdoor defense without considering the diverse range of defense mechanisms. Based on this observation, we pose a natural question: Are current backdoor attacks truly a real-world threat when facing diverse practical defenses?   To answer this question, we examine 12 common backdoor attacks that focus on input-space or feature-space stealthiness and 17 diverse representative defenses. Surprisingly, we reveal a critical blind spot: Backdoor attacks designed to be stealthy in input and feature spaces can be mitigated by examining backdoored models in parameter space. To investigate the underlying causes behind this common vulnerability, we study the characteristics of backdoor attacks in the parameter space. Notably, we find that input- and feature-space attacks introduce prominent backdoor-related neurons in parameter space, which are not thoroughly considered by current backdoor attacks. Taking comprehensive stealthiness into account, we propose a novel supply-chain attack called Grond. Grond limits the parameter changes by a simple yet effective module, Adversarial Backdoor Injection (ABI), which adaptively increases the parameter-space stealthiness during the backdoor injection. Extensive experiments demonstrate that Grond outperforms all 12 backdoor attacks against state-of-the-art (including adaptive) defenses on CIFAR-10, GTSRB, and a subset of ImageNet. In addition, we show that ABI consistently improves the effectiveness of common backdoor attacks.



## **33. Backdoor Attacks against No-Reference Image Quality Assessment Models via a Scalable Trigger**

cs.CV

Accept by AAAI 2025

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2412.07277v2) [paper-pdf](http://arxiv.org/pdf/2412.07277v2)

**Authors**: Yi Yu, Song Xia, Xun Lin, Wenhan Yang, Shijian Lu, Yap-peng Tan, Alex Kot

**Abstract**: No-Reference Image Quality Assessment (NR-IQA), responsible for assessing the quality of a single input image without using any reference, plays a critical role in evaluating and optimizing computer vision systems, e.g., low-light enhancement. Recent research indicates that NR-IQA models are susceptible to adversarial attacks, which can significantly alter predicted scores with visually imperceptible perturbations. Despite revealing vulnerabilities, these attack methods have limitations, including high computational demands, untargeted manipulation, limited practical utility in white-box scenarios, and reduced effectiveness in black-box scenarios. To address these challenges, we shift our focus to another significant threat and present a novel poisoning-based backdoor attack against NR-IQA (BAIQA), allowing the attacker to manipulate the IQA model's output to any desired target value by simply adjusting a scaling coefficient $\alpha$ for the trigger. We propose to inject the trigger in the discrete cosine transform (DCT) domain to improve the local invariance of the trigger for countering trigger diminishment in NR-IQA models due to widely adopted data augmentations. Furthermore, the universal adversarial perturbations (UAP) in the DCT space are designed as the trigger, to increase IQA model susceptibility to manipulation and improve attack effectiveness. In addition to the heuristic method for poison-label BAIQA (P-BAIQA), we explore the design of clean-label BAIQA (C-BAIQA), focusing on $\alpha$ sampling and image data refinement, driven by theoretical insights we reveal. Extensive experiments on diverse datasets and various NR-IQA models demonstrate the effectiveness of our attacks. Code can be found at https://github.com/yuyi-sd/BAIQA.



## **34. Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs**

cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.01042v2) [paper-pdf](http://arxiv.org/pdf/2501.01042v2)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models--a common and practical real world scenario--remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal model (IMM) as a surrogate model to craft adversarial video samples. Multimodal interactions and temporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. In addition, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as surrogate model) achieve competitive performance, with average attack success rates of 55.48% on MSVD-QA and 58.26% on MSRVTT-QA for VideoQA tasks, respectively. Our code will be released upon acceptance.



## **35. ActMiner: Applying Causality Tracking and Increment Aligning for Graph-based Cyber Threat Hunting**

cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05793v1) [paper-pdf](http://arxiv.org/pdf/2501.05793v1)

**Authors**: Mingjun Ma, Tiantian Zhu, Tieming Chen, Shuang Li, Jie Ying, Chunlin Xiong, Mingqi Lv, Yan Chen

**Abstract**: To defend against Advanced Persistent Threats on the endpoint, threat hunting employs security knowledge such as cyber threat intelligence to continuously analyze system audit logs through retrospective scanning, querying, or pattern matching, aiming to uncover attack patterns/graphs that traditional detection methods (e.g., recognition for Point of Interest) fail to capture. However, existing threat hunting systems based on provenance graphs face challenges of high false negatives, high false positives, and low efficiency when confronted with diverse attack tactics and voluminous audit logs. To address these issues, we propose a system called Actminer, which constructs query graphs from descriptive relationships in cyber threat intelligence reports for precise threat hunting (i.e., graph alignment) on provenance graphs. First, we present a heuristic search strategy based on equivalent semantic transfer to reduce false negatives. Second, we establish a filtering mechanism based on causal relationships of attack behaviors to mitigate false positives. Finally, we design a tree structure to incrementally update the alignment results, significantly improving hunting efficiency. Evaluation on the DARPA Engagement dataset demonstrates that compared to the SOTA POIROT, Actminer reduces false positives by 39.1%, eliminates all false negatives, and effectively counters adversarial attacks.



## **36. UV-Attack: Physical-World Adversarial Attacks for Person Detection via Dynamic-NeRF-based UV Mapping**

cs.CV

23 pages, 22 figures, submitted to ICLR2025

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05783v1) [paper-pdf](http://arxiv.org/pdf/2501.05783v1)

**Authors**: Yanjie Li, Wenxuan Zhang, Kaisheng Liang, Bin Xiao

**Abstract**: In recent research, adversarial attacks on person detectors using patches or static 3D model-based texture modifications have struggled with low success rates due to the flexible nature of human movement. Modeling the 3D deformations caused by various actions has been a major challenge. Fortunately, advancements in Neural Radiance Fields (NeRF) for dynamic human modeling offer new possibilities. In this paper, we introduce UV-Attack, a groundbreaking approach that achieves high success rates even with extensive and unseen human actions. We address the challenge above by leveraging dynamic-NeRF-based UV mapping. UV-Attack can generate human images across diverse actions and viewpoints, and even create novel actions by sampling from the SMPL parameter space. While dynamic NeRF models are capable of modeling human bodies, modifying clothing textures is challenging because they are embedded in neural network parameters. To tackle this, UV-Attack generates UV maps instead of RGB images and modifies the texture stacks. This approach enables real-time texture edits and makes the attack more practical. We also propose a novel Expectation over Pose Transformation loss (EoPT) to improve the evasion success rate on unseen poses and views. Our experiments show that UV-Attack achieves a 92.75% attack success rate against the FastRCNN model across varied poses in dynamic video settings, significantly outperforming the state-of-the-art AdvCamou attack, which only had a 28.50% ASR. Moreover, we achieve 49.5% ASR on the latest YOLOv8 detector in black-box settings. This work highlights the potential of dynamic NeRF-based UV mapping for creating more effective adversarial attacks on person detectors, addressing key challenges in modeling human movement and texture modification.



## **37. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.09093v2) [paper-pdf](http://arxiv.org/pdf/2408.09093v2)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **38. An Efficiency Firmware Verification Framework for Public Key Infrastructure with Smart Grid and Energy Storage System**

cs.CR

10pages, 5 figures

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05722v1) [paper-pdf](http://arxiv.org/pdf/2501.05722v1)

**Authors**: Jhih-Zen Shih, Cheng-Che Chuang, Hong-Sheng Huang, Hsuan-Tung Chen, Hung-Min Sun

**Abstract**: As a critical component of electrical energy infrastructure, the smart grid system has become indispensable to the energy sector. However, the rapid evolution of smart grids has attracted numerous nation-state actors seeking to disrupt the power infrastructure of adversarial nations. This development underscores the urgent need to establish secure mechanisms for firmware updates, with firmware signing and verification serving as pivotal elements in safeguarding system integrity. In this work, we propose a digital signing and verification framework grounded in Public Key Infrastructure (PKI), specifically tailored for resource-constrained devices such as smart meters. The framework utilizes the Concise Binary Object Representation (CBOR) and Object Signing and Encryption (COSE) formats to achieve efficient da-ta encapsulation and robust security features. Our approach not only en-sures the secure deployment of firmware updates against the convergence of information technology (IT) and operational technology (OT) attacks but also addresses performance bottlenecks stemming from device limitations, thereby enhancing the overall reliability and stability of the smart grid sys-tem.



## **39. Adversarial Robustness for Deep Learning-based Wildfire Prediction Models**

cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2412.20006v2) [paper-pdf](http://arxiv.org/pdf/2412.20006v2)

**Authors**: Ryo Ide, Lei Yang

**Abstract**: Smoke detection using Deep Neural Networks (DNNs) is an effective approach for early wildfire detection. However, because smoke is temporally and spatially anomalous, there are limitations in collecting sufficient training data. This raises overfitting and bias concerns in existing DNN-based wildfire detection models. Thus, we introduce WARP (Wildfire Adversarial Robustness Procedure), the first model-agnostic framework for evaluating the adversarial robustness of DNN-based wildfire detection models. WARP addresses limitations in smoke image diversity using global and local adversarial attack methods. The global attack method uses image-contextualized Gaussian noise, while the local attack method uses patch noise injection, tailored to address critical aspects of wildfire detection. Leveraging WARP's model-agnostic capabilities, we assess the adversarial robustness of real-time Convolutional Neural Networks (CNNs) and Transformers. The analysis revealed valuable insights into the models' limitations. Specifically, the global attack method demonstrates that the Transformer model has more than 70% precision degradation than the CNN against global noise. In contrast, the local attack method shows that both models are susceptible to cloud image injections when detecting smoke-positive instances, suggesting a need for model improvements through data augmentation. WARP's comprehensive robustness analysis contributed to the development of wildfire-specific data augmentation strategies, marking a step toward practicality.



## **40. Enforcing Fundamental Relations via Adversarial Attacks on Input Parameter Correlations**

cs.LG

12 pages, 8 figures (Without appendix)

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05588v1) [paper-pdf](http://arxiv.org/pdf/2501.05588v1)

**Authors**: Timo Saala, Lucie Flek, Alexander Jung, Akbar Karimi, Alexander Schmidt, Matthias Schott, Philipp Soldin, Christopher Wiebusch

**Abstract**: Correlations between input parameters play a crucial role in many scientific classification tasks, since these are often related to fundamental laws of nature. For example, in high energy physics, one of the common deep learning use-cases is the classification of signal and background processes in particle collisions. In many such cases, the fundamental principles of the correlations between observables are often better understood than the actual distributions of the observables themselves. In this work, we present a new adversarial attack algorithm called Random Distribution Shuffle Attack (RDSA), emphasizing the correlations between observables in the network rather than individual feature characteristics. Correct application of the proposed novel attack can result in a significant improvement in classification performance - particularly in the context of data augmentation - when using the generated adversaries within adversarial training. Given that correlations between input features are also crucial in many other disciplines. We demonstrate the RDSA effectiveness on six classification tasks, including two particle collision challenges (using CERN Open Data), hand-written digit recognition (MNIST784), human activity recognition (HAR), weather forecasting (Rain in Australia), and ICU patient mortality (MIMIC-IV), demonstrating a general use case beyond fundamental physics for this new type of adversarial attack algorithms.



## **41. CROPS: Model-Agnostic Training-Free Framework for Safe Image Synthesis with Latent Diffusion Models**

cs.CV

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05359v1) [paper-pdf](http://arxiv.org/pdf/2501.05359v1)

**Authors**: Junha Park, Ian Ryu, Jaehui Hwang, Hyungkeun Park, Jiyoon Kim, Jong-Seok Lee

**Abstract**: With advances in diffusion models, image generation has shown significant performance improvements. This raises concerns about the potential abuse of image generation, such as the creation of explicit or violent images, commonly referred to as Not Safe For Work (NSFW) content. To address this, the Stable Diffusion model includes several safety checkers to censor initial text prompts and final output images generated from the model. However, recent research has shown that these safety checkers have vulnerabilities against adversarial attacks, allowing them to generate NSFW images. In this paper, we find that these adversarial attacks are not robust to small changes in text prompts or input latents. Based on this, we propose CROPS (Circular or RandOm Prompts for Safety), a model-agnostic framework that easily defends against adversarial attacks generating NSFW images without requiring additional training. Moreover, we develop an approach that utilizes one-step diffusion models for efficient NSFW detection (CROPS-1), further reducing computational resources. We demonstrate the superiority of our method in terms of performance and applicability.



## **42. HPAC-IDS: A Hierarchical Packet Attention Convolution for Intrusion Detection System**

cs.CR

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.06264v1) [paper-pdf](http://arxiv.org/pdf/2501.06264v1)

**Authors**: Anass Grini, Btissam El Khamlichi, Abdellatif El Afia, Amal El Fallah-Seghrouchni

**Abstract**: This research introduces a robust detection system against malicious network traffic, leveraging hierarchical structures and self-attention mechanisms. The proposed system includes a Packet Segmenter that divides a given raw network packet into fixed-size segments that are fed to the HPAC-IDS. The experiments performed on CIC-IDS2017 dataset show that the system exhibits high accuracy and low false positive rates while demonstrating resilience against diverse adversarial methods like Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and Wasserstein GAN (WGAN). The model's ability to withstand adversarial perturbations is attributed to the fusion of hierarchical attention mechanisms and convolutional neural networks, resulting in a 0% to 10% adversarial attack severity under tested adversarial attacks with different segment sizes, surpassing the state-of-the-art model in detection performance and adversarial attack robustness.



## **43. Safeguarding System Prompts for LLMs**

cs.CR

15 pages, 5 figures, 2 tables

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2412.13426v2) [paper-pdf](http://arxiv.org/pdf/2412.13426v2)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a robust defense mechanism designed to safeguard system prompts. PromptKeeper tackles two core challenges: reliably detecting prompt leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon detection, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.



## **44. RAG-WM: An Efficient Black-Box Watermarking Approach for Retrieval-Augmented Generation of Large Language Models**

cs.CR

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05249v1) [paper-pdf](http://arxiv.org/pdf/2501.05249v1)

**Authors**: Peizhuo Lv, Mengjie Sun, Hao Wang, Xiaofeng Wang, Shengzhi Zhang, Yuxuan Chen, Kai Chen, Limin Sun

**Abstract**: In recent years, tremendous success has been witnessed in Retrieval-Augmented Generation (RAG), widely used to enhance Large Language Models (LLMs) in domain-specific, knowledge-intensive, and privacy-sensitive tasks. However, attackers may steal those valuable RAGs and deploy or commercialize them, making it essential to detect Intellectual Property (IP) infringement. Most existing ownership protection solutions, such as watermarks, are designed for relational databases and texts. They cannot be directly applied to RAGs because relational database watermarks require white-box access to detect IP infringement, which is unrealistic for the knowledge base in RAGs. Meanwhile, post-processing by the adversary's deployed LLMs typically destructs text watermark information. To address those problems, we propose a novel black-box "knowledge watermark" approach, named RAG-WM, to detect IP infringement of RAGs. RAG-WM uses a multi-LLM interaction framework, comprising a Watermark Generator, Shadow LLM & RAG, and Watermark Discriminator, to create watermark texts based on watermark entity-relationship tuples and inject them into the target RAG. We evaluate RAG-WM across three domain-specific and two privacy-sensitive tasks on four benchmark LLMs. Experimental results show that RAG-WM effectively detects the stolen RAGs in various deployed LLMs. Furthermore, RAG-WM is robust against paraphrasing, unrelated content removal, knowledge insertion, and knowledge expansion attacks. Lastly, RAG-WM can also evade watermark detection approaches, highlighting its promising application in detecting IP infringement of RAG systems.



## **45. DiffAttack: Diffusion-based Timbre-reserved Adversarial Attack in Speaker Identification**

cs.SD

5 pages,4 figures, accepted by ICASSP 2025

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05127v1) [paper-pdf](http://arxiv.org/pdf/2501.05127v1)

**Authors**: Qing Wang, Jixun Yao, Zhaokai Sun, Pengcheng Guo, Lei Xie, John H. L. Hansen

**Abstract**: Being a form of biometric identification, the security of the speaker identification (SID) system is of utmost importance. To better understand the robustness of SID systems, we aim to perform more realistic attacks in SID, which are challenging for both humans and machines to detect. In this study, we propose DiffAttack, a novel timbre-reserved adversarial attack approach that exploits the capability of a diffusion-based voice conversion (DiffVC) model to generate adversarial fake audio with distinct target speaker attribution. By introducing adversarial constraints into the generative process of the diffusion-based voice conversion model, we craft fake samples that effectively mislead target models while preserving speaker-wise characteristics. Specifically, inspired by the use of randomly sampled Gaussian noise in conventional adversarial attacks and diffusion processes, we incorporate adversarial constraints into the reverse diffusion process. These constraints subtly guide the reverse diffusion process toward aligning with the target speaker distribution. Our experiments on the LibriTTS dataset indicate that DiffAttack significantly improves the attack success rate compared to vanilla DiffVC and other methods. Moreover, objective and subjective evaluations demonstrate that introducing adversarial constraints does not compromise the speech quality generated by the DiffVC model.



## **46. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

cs.CL

Our code is publicly available at  https://github.com/UKPLab/POATE-attack

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.01872v2) [paper-pdf](http://arxiv.org/pdf/2501.01872v2)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.



## **47. On Measuring Unnoticeability of Graph Adversarial Attacks: Observations, New Measure, and Applications**

cs.LG

KDD 2025

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05015v1) [paper-pdf](http://arxiv.org/pdf/2501.05015v1)

**Authors**: Hyeonsoo Jo, Hyunjin Hwang, Fanchen Bu, Soo Yong Lee, Chanyoung Park, Kijung Shin

**Abstract**: Adversarial attacks are allegedly unnoticeable. Prior studies have designed attack noticeability measures on graphs, primarily using statistical tests to compare the topology of original and (possibly) attacked graphs. However, we observe two critical limitations in the existing measures. First, because the measures rely on simple rules, attackers can readily enhance their attacks to bypass them, reducing their attack "noticeability" and, yet, maintaining their attack performance. Second, because the measures naively leverage global statistics, such as degree distributions, they may entirely overlook attacks until severe perturbations occur, letting the attacks be almost "totally unnoticeable." To address the limitations, we introduce HideNSeek, a learnable measure for graph attack noticeability. First, to mitigate the bypass problem, HideNSeek learns to distinguish the original and (potential) attack edges using a learnable edge scorer (LEO), which scores each edge on its likelihood of being an attack. Second, to mitigate the overlooking problem, HideNSeek conducts imbalance-aware aggregation of all the edge scores to obtain the final noticeability score. Using six real-world graphs, we empirically demonstrate that HideNSeek effectively alleviates the observed limitations, and LEO (i.e., our learnable edge scorer) outperforms eleven competitors in distinguishing attack edges under five different attack methods. For an additional application, we show that LEO boost the performance of robust GNNs by removing attack-like edges.



## **48. SpaLLM-Guard: Pairing SMS Spam Detection Using Open-source and Commercial LLMs**

cs.CR

17 pages

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04985v1) [paper-pdf](http://arxiv.org/pdf/2501.04985v1)

**Authors**: Muhammad Salman, Muhammad Ikram, Nardine Basta, Mohamed Ali Kaafar

**Abstract**: The increasing threat of SMS spam, driven by evolving adversarial techniques and concept drift, calls for more robust and adaptive detection methods. In this paper, we evaluate the potential of large language models (LLMs), both open-source and commercial, for SMS spam detection, comparing their performance across zero-shot, few-shot, fine-tuning, and chain-of-thought prompting approaches. Using a comprehensive dataset of SMS messages, we assess the spam detection capabilities of prominent LLMs such as GPT-4, DeepSeek, LLAMA-2, and Mixtral. Our findings reveal that while zero-shot learning provides convenience, it is unreliable for effective spam detection. Few-shot learning, particularly with carefully selected examples, improves detection but exhibits variability across models. Fine-tuning emerges as the most effective strategy, with Mixtral achieving 98.6% accuracy and a balanced false positive and false negative rate below 2%, meeting the criteria for robust spam detection. Furthermore, we explore the resilience of these models to adversarial attacks, finding that fine-tuning significantly enhances robustness against both perceptible and imperceptible manipulations. Lastly, we investigate the impact of concept drift and demonstrate that fine-tuned LLMs, especially when combined with few-shot learning, can mitigate its effects, maintaining high performance even on evolving spam datasets. This study highlights the importance of fine-tuning and tailored learning strategies to deploy LLMs effectively for real-world SMS spam detection



## **49. Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval**

cs.IR

This paper has been accepted for oral presentation in the  reproducibility track at ECIR 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04802v1) [paper-pdf](http://arxiv.org/pdf/2501.04802v1)

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Evangelos Kanoulas

**Abstract**: HotFlip is a topical gradient-based word substitution method for attacking language models. Recently, this method has been further applied to attack retrieval systems by generating malicious passages that are injected into a corpus, i.e., corpus poisoning. However, HotFlip is known to be computationally inefficient, with the majority of time being spent on gradient accumulation for each query-passage pair during the adversarial token generation phase, making it impossible to generate an adequate number of adversarial passages in a reasonable amount of time. Moreover, the attack method itself assumes access to a set of user queries, a strong assumption that does not correspond to how real-world adversarial attacks are usually performed. In this paper, we first significantly boost the efficiency of HotFlip, reducing the adversarial generation process from 4 hours per document to only 15 minutes, using the same hardware. We further contribute experiments and analysis on two additional tasks: (1) transfer-based black-box attacks, and (2) query-agnostic attacks. Whenever possible, we provide comparisons between the original method and our improved version. Our experiments demonstrate that HotFlip can effectively attack a variety of dense retrievers, with an observed trend that its attack performance diminishes against more advanced and recent methods. Interestingly, we observe that while HotFlip performs poorly in a black-box setting, indicating limited capacity for generalization, in query-agnostic scenarios its performance is correlated to the volume of injected adversarial passages.



## **50. Correlated Privacy Mechanisms for Differentially Private Distributed Mean Estimation**

cs.IT

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2407.03289v2) [paper-pdf](http://arxiv.org/pdf/2407.03289v2)

**Authors**: Sajani Vithana, Viveck R. Cadambe, Flavio P. Calmon, Haewon Jeong

**Abstract**: Differentially private distributed mean estimation (DP-DME) is a fundamental building block in privacy-preserving federated learning, where a central server estimates the mean of $d$-dimensional vectors held by $n$ users while ensuring $(\epsilon,\delta)$-DP. Local differential privacy (LDP) and distributed DP with secure aggregation (SA) are the most common notions of DP used in DP-DME settings with an untrusted server. LDP provides strong resilience to dropouts, colluding users, and adversarial attacks, but suffers from poor utility. In contrast, SA-based DP-DME achieves an $O(n)$ utility gain over LDP in DME, but requires increased communication and computation overheads and complex multi-round protocols to handle dropouts and attacks. In this work, we present a generalized framework for DP-DME, that captures LDP and SA-based mechanisms as extreme cases. Our framework provides a foundation for developing and analyzing a variety of DP-DME protocols that leverage correlated privacy mechanisms across users. To this end, we propose CorDP-DME, a novel DP-DME mechanism based on the correlated Gaussian mechanism, that spans the gap between DME with LDP and distributed DP. We prove that CorDP-DME offers a favorable balance between utility and resilience to dropout and collusion. We provide an information-theoretic analysis of CorDP-DME, and derive theoretical guarantees for utility under any given privacy parameters and dropout/colluding user thresholds. Our results demonstrate that (anti) correlated Gaussian DP mechanisms can significantly improve utility in mean estimation tasks compared to LDP -- even in adversarial settings -- while maintaining better resilience to dropouts and attacks compared to distributed DP.



