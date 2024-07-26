# Latest Adversarial Attack Papers
**update at 2024-07-26 10:34:19**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Sparse vs Contiguous Adversarial Pixel Perturbations in Multimodal Models: An Empirical Analysis**

cs.CV

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.18251v1) [paper-pdf](http://arxiv.org/pdf/2407.18251v1)

**Authors**: Cristian-Alexandru Botocan, Raphael Meier, Ljiljana Dolamic

**Abstract**: Assessing the robustness of multimodal models against adversarial examples is an important aspect for the safety of its users. We craft L0-norm perturbation attacks on the preprocessed input images. We launch them in a black-box setup against four multimodal models and two unimodal DNNs, considering both targeted and untargeted misclassification. Our attacks target less than 0.04% of perturbed image area and integrate different spatial positioning of perturbed pixels: sparse positioning and pixels arranged in different contiguous shapes (row, column, diagonal, and patch). To the best of our knowledge, we are the first to assess the robustness of three state-of-the-art multimodal models (ALIGN, AltCLIP, GroupViT) against different sparse and contiguous pixel distribution perturbations. The obtained results indicate that unimodal DNNs are more robust than multimodal models. Furthermore, models using CNN-based Image Encoder are more vulnerable than models with ViT - for untargeted attacks, we obtain a 99% success rate by perturbing less than 0.02% of the image area.



## **2. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2312.03853v4) [paper-pdf](http://arxiv.org/pdf/2312.03853v4)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Gemini (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.



## **3. RIDA: A Robust Attack Framework on Incomplete Graphs**

cs.LG

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.18170v1) [paper-pdf](http://arxiv.org/pdf/2407.18170v1)

**Authors**: Jianke Yu, Hanchen Wang, Chen Chen, Xiaoyang Wang, Wenjie Zhang, Ying Zhang

**Abstract**: Graph Neural Networks (GNNs) are vital in data science but are increasingly susceptible to adversarial attacks. To help researchers develop more robust GNN models, it's essential to focus on designing strong attack models as foundational benchmarks and guiding references. Among adversarial attacks, gray-box poisoning attacks are noteworthy due to their effectiveness and fewer constraints. These attacks exploit GNNs' need for retraining on updated data, thereby impacting their performance by perturbing these datasets. However, current research overlooks the real-world scenario of incomplete graphs.To address this gap, we introduce the Robust Incomplete Deep Attack Framework (RIDA). It is the first algorithm for robust gray-box poisoning attacks on incomplete graphs. The approach innovatively aggregates distant vertex information and ensures powerful data utilization.Extensive tests against 9 SOTA baselines on 3 real-world datasets demonstrate RIDA's superiority in handling incompleteness and high attack performance on the incomplete graph.



## **4. Understanding the Security Benefits and Overheads of Emerging Industry Solutions to DRAM Read Disturbance**

cs.CR

To appear in DRAMSec 2024

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2406.19094v2) [paper-pdf](http://arxiv.org/pdf/2406.19094v2)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Oğuz Ergin, Onur Mutlu

**Abstract**: We present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC), described in JEDEC DDR5 specification's April 2024 update. Unlike prior state-of-the-art that advises the memory controller to periodically issue refresh management (RFM) commands, which provides the DRAM chip with time to perform refreshes, PRAC introduces a new back-off signal. PRAC's back-off signal propagates from the DRAM chip to the memory controller and forces the memory controller to 1) stop serving requests and 2) issue RFM commands. As a result, RFM commands are issued when needed as opposed to periodically, reducing RFM's overheads. We analyze PRAC in four steps. First, we define an adversarial access pattern that represents the worst-case for PRAC's security. Second, we investigate PRAC's configurations and security implications. Our analyses show that PRAC can be configured for secure operation as long as no bitflip occurs before accessing a memory location 10 times. Third, we evaluate the performance impact of PRAC and compare it against prior works using Ramulator 2.0. Our analysis shows that while PRAC incurs less than 13% performance overhead for today's DRAM chips, its performance overheads can reach up to 94% for future DRAM chips that are more vulnerable to read disturbance bitflips. Fourth, we define an availability adversarial access pattern that exacerbates PRAC's performance overhead to perform a memory performance attack, demonstrating that such an adversarial pattern can hog up to 94% of DRAM throughput and degrade system throughput by up to 95%. We discuss PRAC's implications on future systems and foreshadow future research directions. To aid future research, we open-source our implementations and scripts at https://github.com/CMU-SAFARI/ramulator2.



## **5. Chernoff Information as a Privacy Constraint for Adversarial Classification**

cs.IT

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2403.10307v2) [paper-pdf](http://arxiv.org/pdf/2403.10307v2)

**Authors**: Ayşe Ünsal, Melek Önen

**Abstract**: This work inspects a privacy metric based on Chernoff information, \textit{Chernoff differential privacy}, due to its significance in characterization of the optimal classifier's performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we focus on the Bayesian setting and characterize the relationship between the best error exponent of the average error probability and $\varepsilon\textrm{-}$differential privacy \cite{D06}. Accordingly, we re-derive Chernoff differential privacy in terms of $\varepsilon\textrm{-}$differential privacy using the Radon-Nikodym derivative and show that it satisfies the composition property for sequential composition. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$, the impact of the adversary's attack and global sensitivity for the problem of adversarial classification in Laplace mechanisms.



## **6. Is the Digital Forensics and Incident Response Pipeline Ready for Text-Based Threats in LLM Era?**

cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17870v1) [paper-pdf](http://arxiv.org/pdf/2407.17870v1)

**Authors**: Avanti Bhandarkar, Ronald Wilson, Anushka Swarup, Mengdi Zhu, Damon Woodard

**Abstract**: In the era of generative AI, the widespread adoption of Neural Text Generators (NTGs) presents new cybersecurity challenges, particularly within the realms of Digital Forensics and Incident Response (DFIR). These challenges primarily involve the detection and attribution of sources behind advanced attacks like spearphishing and disinformation campaigns. As NTGs evolve, the task of distinguishing between human and NTG-authored texts becomes critically complex. This paper rigorously evaluates the DFIR pipeline tailored for text-based security systems, specifically focusing on the challenges of detecting and attributing authorship of NTG-authored texts. By introducing a novel human-NTG co-authorship text attack, termed CS-ACT, our study uncovers significant vulnerabilities in traditional DFIR methodologies, highlighting discrepancies between ideal scenarios and real-world conditions. Utilizing 14 diverse datasets and 43 unique NTGs, up to the latest GPT-4, our research identifies substantial vulnerabilities in the forensic profiling phase, particularly in attributing authorship to NTGs. Our comprehensive evaluation points to factors such as model sophistication and the lack of distinctive style within NTGs as significant contributors for these vulnerabilities. Our findings underscore the necessity for more sophisticated and adaptable strategies, such as incorporating adversarial learning, stylizing NTGs, and implementing hierarchical attribution through the mapping of NTG lineages to enhance source attribution. This sets the stage for future research and the development of more resilient text-based security systems.



## **7. Domain Generalized Recaptured Screen Image Identification Using SWIN Transformer**

cs.CV

11 pages, 10 figures, 9 tables

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17170v2) [paper-pdf](http://arxiv.org/pdf/2407.17170v2)

**Authors**: Preeti Mehta, Aman Sagar, Suchi Kumari

**Abstract**: An increasing number of classification approaches have been developed to address the issue of image rebroadcast and recapturing, a standard attack strategy in insurance frauds, face spoofing, and video piracy. However, most of them neglected scale variations and domain generalization scenarios, performing poorly in instances involving domain shifts, typically made worse by inter-domain and cross-domain scale variances. To overcome these issues, we propose a cascaded data augmentation and SWIN transformer domain generalization framework (DAST-DG) in the current research work Initially, we examine the disparity in dataset representation. A feature generator is trained to make authentic images from various domains indistinguishable. This process is then applied to recaptured images, creating a dual adversarial learning setup. Extensive experiments demonstrate that our approach is practical and surpasses state-of-the-art methods across different databases. Our model achieves an accuracy of approximately 82\% with a precision of 95\% on high-variance datasets.



## **8. A Unified Understanding of Adversarial Vulnerability Regarding Unimodal Models and Vision-Language Pre-training Models**

cs.CV

14 pages, 9 figures, published in ACMMM2024(oral)

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17797v1) [paper-pdf](http://arxiv.org/pdf/2407.17797v1)

**Authors**: Haonan Zheng, Xinyang Deng, Wen Jiang, Wenrui Li

**Abstract**: With Vision-Language Pre-training (VLP) models demonstrating powerful multimodal interaction capabilities, the application scenarios of neural networks are no longer confined to unimodal domains but have expanded to more complex multimodal V+L downstream tasks. The security vulnerabilities of unimodal models have been extensively examined, whereas those of VLP models remain challenging. We note that in CV models, the understanding of images comes from annotated information, while VLP models are designed to learn image representations directly from raw text. Motivated by this discrepancy, we developed the Feature Guidance Attack (FGA), a novel method that uses text representations to direct the perturbation of clean images, resulting in the generation of adversarial images. FGA is orthogonal to many advanced attack strategies in the unimodal domain, facilitating the direct application of rich research findings from the unimodal to the multimodal scenario. By appropriately introducing text attack into FGA, we construct Feature Guidance with Text Attack (FGA-T). Through the interaction of attacking two modalities, FGA-T achieves superior attack effects against VLP models. Moreover, incorporating data augmentation and momentum mechanisms significantly improves the black-box transferability of FGA-T. Our method demonstrates stable and effective attack capabilities across various datasets, downstream tasks, and both black-box and white-box settings, offering a unified baseline for exploring the robustness of VLP models.



## **9. Exploring Semantic Perturbations on Grover**

cs.LG

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2302.00509v2) [paper-pdf](http://arxiv.org/pdf/2302.00509v2)

**Authors**: Ziqing Ji, Pranav Kulkarni, Marko Neskovic, Kevin Nolan, Yan Xu

**Abstract**: With news and information being as easy to access as they currently are, it is more important than ever to ensure that people are not mislead by what they read. Recently, the rise of neural fake news (AI-generated fake news) and its demonstrated effectiveness at fooling humans has prompted the development of models to detect it. One such model is the Grover model, which can both detect neural fake news to prevent it, and generate it to demonstrate how a model could be misused to fool human readers. In this work we explore the Grover model's fake news detection capabilities by performing targeted attacks through perturbations on input news articles. Through this we test Grover's resilience to these adversarial attacks and expose some potential vulnerabilities which should be addressed in further iterations to ensure it can detect all types of fake news accurately.



## **10. Explaining the Model, Protecting Your Data: Revealing and Mitigating the Data Privacy Risks of Post-Hoc Model Explanations via Membership Inference**

cs.CR

ICML 2024 Workshop on the Next Generation of AI Safety

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17663v1) [paper-pdf](http://arxiv.org/pdf/2407.17663v1)

**Authors**: Catherine Huang, Martin Pawelczyk, Himabindu Lakkaraju

**Abstract**: Predictive machine learning models are becoming increasingly deployed in high-stakes contexts involving sensitive personal data; in these contexts, there is a trade-off between model explainability and data privacy. In this work, we push the boundaries of this trade-off: with a focus on foundation models for image classification fine-tuning, we reveal unforeseen privacy risks of post-hoc model explanations and subsequently offer mitigation strategies for such risks. First, we construct VAR-LRT and L1/L2-LRT, two new membership inference attacks based on feature attribution explanations that are significantly more successful than existing explanation-leveraging attacks, particularly in the low false-positive rate regime that allows an adversary to identify specific training set members with confidence. Second, we find empirically that optimized differentially private fine-tuning substantially diminishes the success of the aforementioned attacks, while maintaining high model accuracy. We carry out a systematic empirical investigation of our 2 new attacks with 5 vision transformer architectures, 5 benchmark datasets, 4 state-of-the-art post-hoc explanation methods, and 4 privacy strength settings.



## **11. Revising the Problem of Partial Labels from the Perspective of CNNs' Robustness**

cs.CV

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17630v1) [paper-pdf](http://arxiv.org/pdf/2407.17630v1)

**Authors**: Xin Zhang, Yuqi Song, Wyatt McCurdy, Xiaofeng Wang, Fei Zuo

**Abstract**: Convolutional neural networks (CNNs) have gained increasing popularity and versatility in recent decades, finding applications in diverse domains. These remarkable achievements are greatly attributed to the support of extensive datasets with precise labels. However, annotating image datasets is intricate and complex, particularly in the case of multi-label datasets. Hence, the concept of partial-label setting has been proposed to reduce annotation costs, and numerous corresponding solutions have been introduced. The evaluation methods for these existing solutions have been primarily based on accuracy. That is, their performance is assessed by their predictive accuracy on the test set. However, we insist that such an evaluation is insufficient and one-sided. On one hand, since the quality of the test set has not been evaluated, the assessment results are unreliable. On the other hand, the partial-label problem may also be raised by undergoing adversarial attacks. Therefore, incorporating robustness into the evaluation system is crucial. For this purpose, we first propose two attack models to generate multiple partial-label datasets with varying degrees of label missing rates. Subsequently, we introduce a lightweight partial-label solution using pseudo-labeling techniques and a designed loss function. Then, we employ D-Score to analyze both the proposed and existing methods to determine whether they can enhance robustness while improving accuracy. Extensive experimental results demonstrate that while certain methods may improve accuracy, the enhancement in robustness is not significant, and in some cases, it even diminishes.



## **12. Fluent Student-Teacher Redteaming**

cs.CL

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17447v1) [paper-pdf](http://arxiv.org/pdf/2407.17447v1)

**Authors**: T. Ben Thompson, Michael Sklar

**Abstract**: Many publicly available language models have been safety tuned to reduce the likelihood of toxic or liability-inducing text. Users or security analysts attempt to jailbreak or redteam these models with adversarial prompts which cause compliance with requests. One attack method is to apply discrete optimization techniques to the prompt. However, the resulting attack strings are often gibberish text, easily filtered by defenders due to high measured perplexity, and may fail for unseen tasks and/or well-tuned models. In this work, we improve existing algorithms (primarily GCG and BEAST) to develop powerful and fluent attacks on safety-tuned models like Llama-2 and Phi-3. Our technique centers around a new distillation-based approach that encourages the victim model to emulate a toxified finetune, either in terms of output probabilities or internal activations. To encourage human-fluent attacks, we add a multi-model perplexity penalty and a repetition penalty to the objective. We also enhance optimizer strength by allowing token insertions, token swaps, and token deletions and by using longer attack sequences. The resulting process is able to reliably jailbreak the most difficult target models with prompts that appear similar to human-written prompts. On Advbench we achieve attack success rates $>93$% for Llama-2-7B, Llama-3-8B, and Vicuna-7B, while maintaining model-measured perplexity $<33$; we achieve $95$% attack success for Phi-3, though with higher perplexity. We also find a universally-optimized single fluent prompt that induces $>88$% compliance on previously unseen tasks across Llama-2-7B, Phi-3-mini and Vicuna-7B and transfers to other black-box models.



## **13. Physical Adversarial Attack on Monocular Depth Estimation via Shape-Varying Patches**

cs.CV

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17312v1) [paper-pdf](http://arxiv.org/pdf/2407.17312v1)

**Authors**: Chenxing Zhao, Yang Li, Shihao Wu, Wenyi Tan, Shuangju Zhou, Quan Pan

**Abstract**: Adversarial attacks against monocular depth estimation (MDE) systems pose significant challenges, particularly in safety-critical applications such as autonomous driving. Existing patch-based adversarial attacks for MDE are confined to the vicinity of the patch, making it difficult to affect the entire target. To address this limitation, we propose a physics-based adversarial attack on monocular depth estimation, employing a framework called Attack with Shape-Varying Patches (ASP), aiming to optimize patch content, shape, and position to maximize effectiveness. We introduce various mask shapes, including quadrilateral, rectangular, and circular masks, to enhance the flexibility and efficiency of the attack. Furthermore, we propose a new loss function to extend the influence of the patch beyond the overlapping regions. Experimental results demonstrate that our attack method generates an average depth error of 18 meters on the target car with a patch area of 1/9, affecting over 98\% of the target area.



## **14. Learning to Transform Dynamically for Better Adversarial Transferability**

cs.CV

accepted as a poster in CVPR 2024

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2405.14077v2) [paper-pdf](http://arxiv.org/pdf/2405.14077v2)

**Authors**: Rongyi Zhu, Zeliang Zhang, Susan Liang, Zhuo Liu, Chenliang Xu

**Abstract**: Adversarial examples, crafted by adding perturbations imperceptible to humans, can deceive neural networks. Recent studies identify the adversarial transferability across various models, \textit{i.e.}, the cross-model attack ability of adversarial samples. To enhance such adversarial transferability, existing input transformation-based methods diversify input data with transformation augmentation. However, their effectiveness is limited by the finite number of available transformations. In our study, we introduce a novel approach named Learning to Transform (L2T). L2T increases the diversity of transformed images by selecting the optimal combination of operations from a pool of candidates, consequently improving adversarial transferability. We conceptualize the selection of optimal transformation combinations as a trajectory optimization problem and employ a reinforcement learning strategy to effectively solve the problem. Comprehensive experiments on the ImageNet dataset, as well as practical tests with Google Vision and GPT-4V, reveal that L2T surpasses current methodologies in enhancing adversarial transferability, thereby confirming its effectiveness and practical significance. The code is available at https://github.com/RongyiZhu/L2T.



## **15. When AI Defeats Password Deception! A Deep Learning Framework to Distinguish Passwords and Honeywords**

cs.CR

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.16964v1) [paper-pdf](http://arxiv.org/pdf/2407.16964v1)

**Authors**: Jimmy Dani, Brandon McCulloh, Nitesh Saxena

**Abstract**: "Honeywords" have emerged as a promising defense mechanism for detecting data breaches and foiling offline dictionary attacks (ODA) by deceiving attackers with false passwords. In this paper, we propose PassFilter, a novel deep learning (DL) based attack framework, fundamental in its ability to identify passwords from a set of sweetwords associated with a user account, effectively challenging a variety of honeywords generation techniques (HGTs). The DL model in PassFilter is trained with a set of previously collected or adversarially generated passwords and honeywords, and carefully orchestrated to predict whether a sweetword is the password or a honeyword. Our model can compromise the security of state-of-the-art, heuristics-based, and representation learning-based HGTs proposed by Dionysiou et al. Specifically, our analysis with nine publicly available password datasets shows that PassFilter significantly outperforms the baseline random guessing success rate of 5%, achieving 6.10% to 52.78% on the 1st guessing attempt, considering 20 sweetwords per account. This success rate rapidly increases with additional login attempts before account lock-outs, often allowed on many real-world online services to maintain reasonable usability. For example, it ranges from 41.78% to 96.80% for five attempts, and from 72.87% to 99.00% for ten attempts, compared to 25% and 50% random guessing, respectively. We also examined PassFilter against general-purpose language models used for honeyword generation, like those proposed by Yu et al. These honeywords also proved vulnerable to our attack, with success rates of 14.19% for 1st guessing attempt, increasing to 30.23%, 41.70%, and 63.10% after 3rd, 5th, and 10th guessing attempts, respectively. Our findings demonstrate the effectiveness of DL model deployed in PassFilter in breaching state-of-the-art HGTs and compromising password security based on ODA.



## **16. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2403.13031v2) [paper-pdf](http://arxiv.org/pdf/2403.13031v2)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.



## **17. RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent**

cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16667v1) [paper-pdf](http://arxiv.org/pdf/2407.16667v1)

**Authors**: Huiyu Xu, Wenhui Zhang, Zhibo Wang, Feng Xiao, Rui Zheng, Yunhe Feng, Zhongjie Ba, Kui Ren

**Abstract**: Recently, advanced Large Language Models (LLMs) such as GPT-4 have been integrated into many real-world applications like Code Copilot. These applications have significantly expanded the attack surface of LLMs, exposing them to a variety of threats. Among them, jailbreak attacks that induce toxic responses through jailbreak prompts have raised critical safety concerns. To identify these threats, a growing number of red teaming approaches simulate potential adversarial scenarios by crafting jailbreak prompts to test the target LLM. However, existing red teaming methods do not consider the unique vulnerabilities of LLM in different scenarios, making it difficult to adjust the jailbreak prompts to find context-specific vulnerabilities. Meanwhile, these methods are limited to refining jailbreak templates using a few mutation operations, lacking the automation and scalability to adapt to different scenarios. To enable context-aware and efficient red teaming, we abstract and model existing attacks into a coherent concept called "jailbreak strategy" and propose a multi-agent LLM system named RedAgent that leverages these strategies to generate context-aware jailbreak prompts. By self-reflecting on contextual feedback in an additional memory buffer, RedAgent continuously learns how to leverage these strategies to achieve effective jailbreaks in specific contexts. Extensive experiments demonstrate that our system can jailbreak most black-box LLMs in just five queries, improving the efficiency of existing red teaming methods by two times. Additionally, RedAgent can jailbreak customized LLM applications more efficiently. By generating context-aware jailbreak prompts towards applications on GPTs, we discover 60 severe vulnerabilities of these real-world applications with only two queries per vulnerability. We have reported all found issues and communicated with OpenAI and Meta for bug fixes.



## **18. S-E Pipeline: A Vision Transformer (ViT) based Resilient Classification Pipeline for Medical Imaging Against Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.17587v1) [paper-pdf](http://arxiv.org/pdf/2407.17587v1)

**Authors**: Neha A S, Vivek Chaturvedi, Muhammad Shafique

**Abstract**: Vision Transformer (ViT) is becoming widely popular in automating accurate disease diagnosis in medical imaging owing to its robust self-attention mechanism. However, ViTs remain vulnerable to adversarial attacks that may thwart the diagnosis process by leading it to intentional misclassification of critical disease. In this paper, we propose a novel image classification pipeline, namely, S-E Pipeline, that performs multiple pre-processing steps that allow ViT to be trained on critical features so as to reduce the impact of input perturbations by adversaries. Our method uses a combination of segmentation and image enhancement techniques such as Contrast Limited Adaptive Histogram Equalization (CLAHE), Unsharp Masking (UM), and High-Frequency Emphasis filtering (HFE) as preprocessing steps to identify critical features that remain intact even after adversarial perturbations. The experimental study demonstrates that our novel pipeline helps in reducing the effect of adversarial attacks by 72.22% for the ViT-b32 model and 86.58% for the ViT-l32 model. Furthermore, we have shown an end-to-end deployment of our proposed method on the NVIDIA Jetson Orin Nano board to demonstrate its practical use case in modern hand-held devices that are usually resource-constrained.



## **19. Defending Our Privacy With Backdoors**

cs.LG

Accepted at ECAI 2024

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2310.08320v4) [paper-pdf](http://arxiv.org/pdf/2310.08320v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information, such as names and faces of individuals, from vision-language models by fine-tuning them for only a few minutes instead of re-training them from scratch. Specifically, by strategically inserting backdoors into text encoders, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's actual name. For image encoders, we map individuals' embeddings to be removed from the model to a universal, anonymous embedding. The results of our extensive experimental evaluation demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides a new "dual-use" perspective on backdoor attacks and presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.



## **20. Securing Tomorrow's Smart Cities: Investigating Software Security in Internet of Vehicles and Deep Learning Technologies**

cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16410v1) [paper-pdf](http://arxiv.org/pdf/2407.16410v1)

**Authors**: Ridhi Jain, Norbert Tihanyi, Mohamed Amine Ferrag

**Abstract**: Integrating Deep Learning (DL) techniques in the Internet of Vehicles (IoV) introduces many security challenges and issues that require thorough examination. This literature review delves into the inherent vulnerabilities and risks associated with DL in IoV systems, shedding light on the multifaceted nature of security threats. Through an extensive analysis of existing research, we explore potential threats posed by DL algorithms, including adversarial attacks, data privacy breaches, and model poisoning. Additionally, we investigate the impact of DL on critical aspects of IoV security, such as intrusion detection, anomaly detection, and secure communication protocols. Our review emphasizes the complexities of ensuring the robustness, reliability, and trustworthiness of DL-based IoV systems, given the dynamic and interconnected nature of vehicular networks. Furthermore, we discuss the need for novel security solutions tailored to address these challenges effectively and enhance the security posture of DL-enabled IoV environments. By offering insights into these critical issues, this chapter aims to stimulate further research, innovation, and collaboration in securing DL techniques within the context of the IoV, thereby fostering a safer and more resilient future for vehicular communication and connectivity.



## **21. Protecting Quantum Procrastinators with Signature Lifting: A Case Study in Cryptocurrencies**

cs.CR

Minor revision

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2303.06754v2) [paper-pdf](http://arxiv.org/pdf/2303.06754v2)

**Authors**: Or Sattath, Shai Wyborski

**Abstract**: Current solutions to quantum vulnerabilities of widely used cryptographic schemes involve migrating users to post-quantum schemes before quantum attacks become feasible. This work deals with protecting quantum procrastinators: users that failed to migrate to post-quantum cryptography in time.   To address this problem in the context of digital signatures, we introduce a technique called signature lifting, that allows us to lift a deployed pre-quantum signature scheme satisfying a certain property to a post-quantum signature scheme that uses the same keys. Informally, the said property is that a post-quantum one-way function is used "somewhere along the way" to derive the public-key from the secret-key. Our constructions of signature lifting relies heavily on the post-quantum digital signature scheme Picnic (Chase et al., CCS'17).   Our main case-study is cryptocurrencies, where this property holds in two scenarios: when the public-key is generated via a key-derivation function or when the public-key hash is posted instead of the public-key itself. We propose a modification, based on signature lifting, that can be applied in many cryptocurrencies for securely spending pre-quantum coins in presence of quantum adversaries. Our construction improves upon existing constructions in two major ways: it is not limited to pre-quantum coins whose ECDSA public-key has been kept secret (and in particular, it handles all coins that are stored in addresses generated by HD wallets), and it does not require access to post-quantum coins or using side payments to pay for posting the transaction.



## **22. Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models**

cs.CV

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2404.10335v3) [paper-pdf](http://arxiv.org/pdf/2404.10335v3)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo

**Abstract**: Adversarial attacks, particularly \textbf{targeted} transfer-based attacks, can be used to assess the adversarial robustness of large visual-language models (VLMs), allowing for a more thorough examination of potential security flaws before deployment. However, previous transfer-based adversarial attacks incur high costs due to high iteration counts and complex method structure. Furthermore, due to the unnaturalness of adversarial semantics, the generated adversarial examples have low transferability. These issues limit the utility of existing methods for assessing robustness. To address these issues, we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask method to disperse adversarial semantics throughout the image rather than concentrating them in a single area. Finally, AdvDiffVLM embeds more target semantics into adversarial examples after multiple iterations. Experimental results show that our method generates adversarial examples 5x to 10x faster than state-of-the-art transfer-based adversarial attacks while maintaining higher quality adversarial examples. Furthermore, compared to previous transfer-based adversarial attacks, the adversarial examples generated by our method have better transferability. Notably, AdvDiffVLM can successfully attack a variety of commercial VLMs in a black-box environment, including GPT-4V.



## **23. R.A.C.E.: Robust Adversarial Concept Erasure for Secure Text-to-Image Diffusion Model**

cs.CV

Accepted at ECCV 2024

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2405.16341v2) [paper-pdf](http://arxiv.org/pdf/2405.16341v2)

**Authors**: Changhoon Kim, Kyle Min, Yezhou Yang

**Abstract**: In the evolving landscape of text-to-image (T2I) diffusion models, the remarkable capability to generate high-quality images from textual descriptions faces challenges with the potential misuse of reproducing sensitive content. To address this critical issue, we introduce \textbf{R}obust \textbf{A}dversarial \textbf{C}oncept \textbf{E}rase (RACE), a novel approach designed to mitigate these risks by enhancing the robustness of concept erasure method for T2I models. RACE utilizes a sophisticated adversarial training framework to identify and mitigate adversarial text embeddings, significantly reducing the Attack Success Rate (ASR). Impressively, RACE achieves a 30 percentage point reduction in ASR for the ``nudity'' concept against the leading white-box attack method. Our extensive evaluations demonstrate RACE's effectiveness in defending against both white-box and black-box attacks, marking a significant advancement in protecting T2I diffusion models from generating inappropriate or misleading imagery. This work underlines the essential need for proactive defense measures in adapting to the rapidly advancing field of adversarial challenges. Our code is publicly available: \url{https://github.com/chkimmmmm/R.A.C.E.}



## **24. Algebraic Adversarial Attacks on Integrated Gradients**

cs.LG

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16233v1) [paper-pdf](http://arxiv.org/pdf/2407.16233v1)

**Authors**: Lachlan Simpson, Federico Costanza, Kyle Millar, Adriel Cheng, Cheng-Chew Lim, Hong Gunn Chew

**Abstract**: Adversarial attacks on explainability models have drastic consequences when explanations are used to understand the reasoning of neural networks in safety critical systems. Path methods are one such class of attribution methods susceptible to adversarial attacks. Adversarial learning is typically phrased as a constrained optimisation problem. In this work, we propose algebraic adversarial examples and study the conditions under which one can generate adversarial examples for integrated gradients. Algebraic adversarial examples provide a mathematically tractable approach to adversarial examples.



## **25. EVD4UAV: An Altitude-Sensitive Benchmark to Evade Vehicle Detection in UAV**

cs.CV

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2403.05422v2) [paper-pdf](http://arxiv.org/pdf/2403.05422v2)

**Authors**: Huiming Sun, Jiacheng Guo, Zibo Meng, Tianyun Zhang, Jianwu Fang, Yuewei Lin, Hongkai Yu

**Abstract**: Vehicle detection in Unmanned Aerial Vehicle (UAV) captured images has wide applications in aerial photography and remote sensing. There are many public benchmark datasets proposed for the vehicle detection and tracking in UAV images. Recent studies show that adding an adversarial patch on objects can fool the well-trained deep neural networks based object detectors, posing security concerns to the downstream tasks. However, the current public UAV datasets might ignore the diverse altitudes, vehicle attributes, fine-grained instance-level annotation in mostly side view with blurred vehicle roof, so none of them is good to study the adversarial patch based vehicle detection attack problem. In this paper, we propose a new dataset named EVD4UAV as an altitude-sensitive benchmark to evade vehicle detection in UAV with 6,284 images and 90,886 fine-grained annotated vehicles. The EVD4UAV dataset has diverse altitudes (50m, 70m, 90m), vehicle attributes (color, type), fine-grained annotation (horizontal and rotated bounding boxes, instance-level mask) in top view with clear vehicle roof. One white-box and two black-box patch based attack methods are implemented to attack three classic deep neural networks based object detectors on EVD4UAV. The experimental results show that these representative attack methods could not achieve the robust altitude-insensitive attack performance.



## **26. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

cs.LG

11 pages, 7 figures, 2 tables, 1 algorithm. Version Update: Figure 6

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2406.18451v2) [paper-pdf](http://arxiv.org/pdf/2406.18451v2)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate strong margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively use the logit margin to confidently detect brittle decisions with such models and accurately estimate robust accuracy on an arbitrarily large test set by estimating the input margins only on a small subset. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to efficiently assess adversarial vulnerability in deployment scenarios.



## **27. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**

cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2402.16822v2) [paper-pdf](http://arxiv.org/pdf/2402.16822v2)

**Authors**: Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu

**Abstract**: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to adversarial attacks is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel black-box approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem, and uses open-ended search to generate prompts that are both effective and diverse. Focusing on the safety domain, we use Rainbow Teaming to target various state-of-the-art LLMs, including the Llama 2 and Llama 3 models. Our approach reveals hundreds of effective adversarial prompts, with an attack success rate exceeding 90% across all tested models. Furthermore, we demonstrate that fine-tuning models with synthetic data generated by the Rainbow Teaming method significantly enhances their safety without sacrificing general performance or helpfulness. We additionally explore the versatility of Rainbow Teaming by applying it to question answering and cybersecurity, showcasing its potential to drive robust open-ended self-improvement in a wide range of applications.



## **28. Enhancing Transferability of Targeted Adversarial Examples: A Self-Universal Perspective**

cs.CV

8 pages and 9 figures

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15683v1) [paper-pdf](http://arxiv.org/pdf/2407.15683v1)

**Authors**: Bowen Peng, Li Liu, Tianpeng Liu, Zhen Liu, Yongxiang Liu

**Abstract**: Transfer-based targeted adversarial attacks against black-box deep neural networks (DNNs) have been proven to be significantly more challenging than untargeted ones. The impressive transferability of current SOTA, the generative methods, comes at the cost of requiring massive amounts of additional data and time-consuming training for each targeted label. This results in limited efficiency and flexibility, significantly hindering their deployment in practical applications. In this paper, we offer a self-universal perspective that unveils the great yet underexplored potential of input transformations in pursuing this goal. Specifically, transformations universalize gradient-based attacks with intrinsic but overlooked semantics inherent within individual images, exhibiting similar scalability and comparable results to time-consuming learning over massive additional data from diverse classes. We also contribute a surprising empirical insight that one of the most fundamental transformations, simple image scaling, is highly effective, scalable, sufficient, and necessary in enhancing targeted transferability. We further augment simple scaling with orthogonal transformations and block-wise applicability, resulting in the Simple, faSt, Self-universal yet Strong Scale Transformation (S$^4$ST) for self-universal TTA. On the ImageNet-Compatible benchmark dataset, our method achieves a 19.8% improvement in the average targeted transfer success rate against various challenging victim models over existing SOTA transformation methods while only consuming 36% time for attacking. It also outperforms resource-intensive attacks by a large margin in various challenging settings.



## **29. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

cs.CL

8 pages

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2406.11260v2) [paper-pdf](http://arxiv.org/pdf/2406.11260v2)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.



## **30. Revisiting the Robust Alignment of Circuit Breakers**

cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15902v1) [paper-pdf](http://arxiv.org/pdf/2407.15902v1)

**Authors**: Leo Schwinn, Simon Geisler

**Abstract**: Over the past decade, adversarial training has emerged as one of the few reliable methods for enhancing model robustness against adversarial attacks [Szegedy et al., 2014, Madry et al., 2018, Xhonneux et al., 2024], while many alternative approaches have failed to withstand rigorous subsequent evaluations. Recently, an alternative defense mechanism, namely "circuit breakers" [Zou et al., 2024], has shown promising results for aligning LLMs. In this report, we show that the robustness claims of "Improving Alignment and Robustness with Circuit Breakers" against unconstraint continuous attacks in the embedding space of the input tokens may be overestimated [Zou et al., 2024]. Specifically, we demonstrate that by implementing a few simple changes to embedding space attacks [Schwinn et al., 2024a,b], we achieve 100% attack success rate (ASR) against circuit breaker models. Without conducting any further hyperparameter tuning, these adjustments increase the ASR by more than 80% compared to the original evaluation. Code is accessible at: https://github.com/SchwinnL/circuit-breakers-eval



## **31. Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

cs.LG

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15549v1) [paper-pdf](http://arxiv.org/pdf/2407.15549v1)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of `jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.



## **32. Towards Efficient Transferable Preemptive Adversarial Defense**

cs.CR

Under Review

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15524v1) [paper-pdf](http://arxiv.org/pdf/2407.15524v1)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy because of its sensitivity to inconspicuous perturbations (i.e., adversarial attacks). Attackers utilize this sensitivity to slightly manipulate transmitted messages. To defend against such attacks, we have devised a strategy for "attacking" the message before it is attacked. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first to our knowledge effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing active defenses against adversarial attacks.



## **33. A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.13863v2) [paper-pdf](http://arxiv.org/pdf/2407.13863v2)

**Authors**: Yixiang Qiu, Hao Fang, Hongyao Yu, Bin Chen, MeiKang Qiu, Shu-Tao Xia

**Abstract**: Model Inversion (MI) attacks aim to reconstruct privacy-sensitive training data from released models by utilizing output information, raising extensive concerns about the security of Deep Neural Networks (DNNs). Recent advances in generative adversarial networks (GANs) have contributed significantly to the improved performance of MI attacks due to their powerful ability to generate realistic images with high fidelity and appropriate semantics. However, previous MI attacks have solely disclosed private information in the latent space of GAN priors, limiting their semantic extraction and transferability across multiple target models and datasets. To address this challenge, we propose a novel method, Intermediate Features enhanced Generative Model Inversion (IF-GMI), which disassembles the GAN structure and exploits features between intermediate blocks. This allows us to extend the optimization space from latent code to intermediate features with enhanced expressive capabilities. To prevent GAN priors from generating unrealistic images, we apply a L1 ball constraint to the optimization process. Experiments on multiple benchmarks demonstrate that our method significantly outperforms previous approaches and achieves state-of-the-art results under various settings, especially in the out-of-distribution (OOD) scenario. Our code is available at: https://github.com/final-solution/IF-GMI



## **34. CLIP-Guided Networks for Transferable Targeted Attacks**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.10179v2) [paper-pdf](http://arxiv.org/pdf/2407.10179v2)

**Authors**: Hao Fang, Jiawei Kong, Bin Chen, Tao Dai, Hao Wu, Shu-Tao Xia

**Abstract**: Transferable targeted adversarial attacks aim to mislead models into outputting adversary-specified predictions in black-box scenarios. Recent studies have introduced \textit{single-target} generative attacks that train a generator for each target class to generate highly transferable perturbations, resulting in substantial computational overhead when handling multiple classes. \textit{Multi-target} attacks address this by training only one class-conditional generator for multiple classes. However, the generator simply uses class labels as conditions, failing to leverage the rich semantic information of the target class. To this end, we design a \textbf{C}LIP-guided \textbf{G}enerative \textbf{N}etwork with \textbf{C}ross-attention modules (CGNC) to enhance multi-target attacks by incorporating textual knowledge of CLIP into the generator. Extensive experiments demonstrate that CGNC yields significant improvements over previous multi-target generative attacks, e.g., a 21.46\% improvement in success rate from ResNet-152 to DenseNet-121. Moreover, we propose a masked fine-tuning mechanism to further strengthen our method in attacking a single class, which surpasses existing single-target methods.



## **35. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.09164v3) [paper-pdf](http://arxiv.org/pdf/2407.09164v3)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate of up to 98.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.



## **36. Imposter.AI: Adversarial Attacks with Hidden Intentions towards Aligned Large Language Models**

cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15399v1) [paper-pdf](http://arxiv.org/pdf/2407.15399v1)

**Authors**: Xiao Liu, Liangzhi Li, Tong Xiang, Fuying Ye, Lu Wei, Wangyue Li, Noa Garcia

**Abstract**: With the development of large language models (LLMs) like ChatGPT, both their vast applications and potential vulnerabilities have come to the forefront. While developers have integrated multiple safety mechanisms to mitigate their misuse, a risk remains, particularly when models encounter adversarial inputs. This study unveils an attack mechanism that capitalizes on human conversation strategies to extract harmful information from LLMs. We delineate three pivotal strategies: (i) decomposing malicious questions into seemingly innocent sub-questions; (ii) rewriting overtly malicious questions into more covert, benign-sounding ones; (iii) enhancing the harmfulness of responses by prompting models for illustrative examples. Unlike conventional methods that target explicit malicious responses, our approach delves deeper into the nature of the information provided in responses. Through our experiments conducted on GPT-3.5-turbo, GPT-4, and Llama2, our method has demonstrated a marked efficacy compared to conventional attack methods. In summary, this work introduces a novel attack method that outperforms previous approaches, raising an important question: How to discern whether the ultimate intent in a dialogue is malicious?



## **37. Towards Robust Vision Transformer via Masked Adaptive Ensemble**

cs.CV

9 pages

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15385v1) [paper-pdf](http://arxiv.org/pdf/2407.15385v1)

**Authors**: Fudong Lin, Jiadong Lou, Xu Yuan, Nian-Feng Tzeng

**Abstract**: Adversarial training (AT) can help improve the robustness of Vision Transformers (ViT) against adversarial attacks by intentionally injecting adversarial examples into the training data. However, this way of adversarial injection inevitably incurs standard accuracy degradation to some extent, thereby calling for a trade-off between standard accuracy and robustness. Besides, the prominent AT solutions are still vulnerable to adaptive attacks. To tackle such shortcomings, this paper proposes a novel ViT architecture, including a detector and a classifier bridged by our newly developed adaptive ensemble. Specifically, we empirically discover that detecting adversarial examples can benefit from the Guided Backpropagation technique. Driven by this discovery, a novel Multi-head Self-Attention (MSA) mechanism is introduced to enhance our detector to sniff adversarial examples. Then, a classifier with two encoders is employed for extracting visual representations respectively from clean images and adversarial examples, with our adaptive ensemble to adaptively adjust the proportion of visual representations from the two encoders for accurate classification. This design enables our ViT architecture to achieve a better trade-off between standard accuracy and robustness. Besides, our adaptive ensemble technique allows us to mask off a random subset of image patches within input data, boosting our ViT's robustness against adaptive attacks, while maintaining high standard accuracy. Experimental results exhibit that our ViT architecture, on CIFAR-10, achieves the best standard accuracy and adversarial robustness of 90.3% and 49.8%, respectively.



## **38. Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing**

cs.LG

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2301.12554v5) [paper-pdf](http://arxiv.org/pdf/2301.12554v5)

**Authors**: Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi

**Abstract**: While prior research has proposed a plethora of methods that build neural classifiers robust against adversarial robustness, practitioners are still reluctant to adopt them due to their unacceptably severe clean accuracy penalties. This paper significantly alleviates this accuracy-robustness trade-off by mixing the output probabilities of a standard classifier and a robust classifier, where the standard network is optimized for clean accuracy and is not robust in general. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key to this improvement. In addition to providing intuitions and empirical evidence, we theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The proposed flexible method, termed "adaptive smoothing", can work in conjunction with existing or even future methods that improve clean accuracy, robustness, or adversary detection. Our empirical evaluation considers strong attack methods, including AutoAttack and adaptive attack. On the CIFAR-100 dataset, our method achieves an 85.21% clean accuracy while maintaining a 38.72% $\ell_\infty$-AutoAttacked ($\epsilon = 8/255$) accuracy, becoming the second most robust method on the RobustBench CIFAR-100 benchmark as of submission, while improving the clean accuracy by ten percentage points compared with all listed models. The code that implements our method is available at https://github.com/Bai-YT/AdaptiveSmoothing.



## **39. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

cs.MM

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2308.12636v3) [paper-pdf](http://arxiv.org/pdf/2308.12636v3)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Hanwang Zhang, Hang Su, Richang Hong

**Abstract**: The integration of visual and textual data in Vision-Language Pre-training (VLP) models is crucial for enhancing vision-language understanding. However, the adversarial robustness of these models, especially in the alignment of image-text features, has not yet been sufficiently explored. In this paper, we introduce a novel gradient-based multimodal adversarial attack method, underpinned by contrastive learning, to improve the transferability of multimodal adversarial samples in VLP models. This method concurrently generates adversarial texts and images within imperceptive perturbation, employing both image-text and intra-modal contrastive loss. We evaluate the effectiveness of our approach on image-text retrieval and visual entailment tasks, using publicly available datasets in a black-box setting. Extensive experiments indicate a significant advancement over existing single-modal transfer-based adversarial attack methods and current multimodal adversarial attack approaches.



## **40. When Do Universal Image Jailbreaks Transfer Between Vision-Language Models?**

cs.CL

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.15211v1) [paper-pdf](http://arxiv.org/pdf/2407.15211v1)

**Authors**: Rylan Schaeffer, Dan Valentine, Luke Bailey, James Chua, Cristóbal Eyzaguirre, Zane Durante, Joe Benton, Brando Miranda, Henry Sleight, John Hughes, Rajashree Agrawal, Mrinank Sharma, Scott Emmons, Sanmi Koyejo, Ethan Perez

**Abstract**: The integration of new modalities into frontier AI systems offers exciting capabilities, but also increases the possibility such systems can be adversarially manipulated in undesirable ways. In this work, we focus on a popular class of vision-language models (VLMs) that generate text outputs conditioned on visual and textual inputs. We conducted a large-scale empirical study to assess the transferability of gradient-based universal image "jailbreaks" using a diverse set of over 40 open-parameter VLMs, including 18 new VLMs that we publicly release. Overall, we find that transferable gradient-based image jailbreaks are extremely difficult to obtain. When an image jailbreak is optimized against a single VLM or against an ensemble of VLMs, the jailbreak successfully jailbreaks the attacked VLM(s), but exhibits little-to-no transfer to any other VLMs; transfer is not affected by whether the attacked and target VLMs possess matching vision backbones or language models, whether the language model underwent instruction-following and/or safety-alignment training, or many other factors. Only two settings display partially successful transfer: between identically-pretrained and identically-initialized VLMs with slightly different VLM training data, and between different training checkpoints of a single VLM. Leveraging these results, we then demonstrate that transfer can be significantly improved against a specific target VLM by attacking larger ensembles of "highly-similar" VLMs. These results stand in stark contrast to existing evidence of universal and transferable text jailbreaks against language models and transferable adversarial attacks against image classifiers, suggesting that VLMs may be more robust to gradient-based transfer attacks.



## **41. SNNGX: Securing Spiking Neural Networks with Genetic XOR Encryption on RRAM-based Neuromorphic Accelerator**

cs.CR

International Conference on Computer-Aided Design 2024

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.15152v1) [paper-pdf](http://arxiv.org/pdf/2407.15152v1)

**Authors**: Kwunhang Wong, Songqi Wang, Wei Huang, Xinyuan Zhang, Yangu He, Karl M. H. Lai, Yuzhong Jiao, Ning Lin, Xiaojuan Qi, Xiaoming Chen, Zhongrui Wang

**Abstract**: Biologically plausible Spiking Neural Networks (SNNs), characterized by spike sparsity, are growing tremendous attention over intellectual edge devices and critical bio-medical applications as compared to artificial neural networks (ANNs). However, there is a considerable risk from malicious attempts to extract white-box information (i.e., weights) from SNNs, as attackers could exploit well-trained SNNs for profit and white-box adversarial concerns. There is a dire need for intellectual property (IP) protective measures. In this paper, we present a novel secure software-hardware co-designed RRAM-based neuromorphic accelerator for protecting the IP of SNNs. Software-wise, we design a tailored genetic algorithm with classic XOR encryption to target the least number of weights that need encryption. From a hardware perspective, we develop a low-energy decryption module, meticulously designed to provide zero decryption latency. Extensive results from various datasets, including NMNIST, DVSGesture, EEGMMIDB, Braille Letter, and SHD, demonstrate that our proposed method effectively secures SNNs by encrypting a minimal fraction of stealthy weights, only 0.00005% to 0.016% weight bits. Additionally, it achieves a substantial reduction in energy consumption, ranging from x59 to x6780, and significantly lowers decryption latency, ranging from x175 to x4250. Moreover, our method requires as little as one sample per class in dataset for encryption and addresses hessian/gradient-based search insensitive problems. This strategy offers a highly efficient and flexible solution for securing SNNs in diverse applications.



## **42. Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models**

cs.CV

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14971v1) [paper-pdf](http://arxiv.org/pdf/2407.14971v1)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Vision-language models (VLMs) have achieved significant strides in recent times specially in multimodal tasks, yet they remain susceptible to adversarial attacks on their vision components. To address this, we propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks while maintaining semantic richness and specificity. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. Our results demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while preserving semantic meaning of the perturbed images. Notably, Sim-CLIP does not require additional training or fine-tuning of the VLM itself; replacing the original vision encoder with our fine-tuned Sim-CLIP suffices to provide robustness. This work underscores the significance of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications, paving the way for more secure and effective multimodal systems.



## **43. Characterizing Encrypted Application Traffic through Cellular Radio Interface Protocol**

cs.NI

9 pages, 8 figures, 2 tables. This paper has been accepted for  publication by the 21st IEEE International Conference on Mobile Ad-Hoc and  Smart Systems (MASS 2024)

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.07361v2) [paper-pdf](http://arxiv.org/pdf/2407.07361v2)

**Authors**: Md Ruman Islam, Raja Hasnain Anwar, Spyridon Mastorakis, Muhammad Taqi Raza

**Abstract**: Modern applications are end-to-end encrypted to prevent data from being read or secretly modified. 5G tech nology provides ubiquitous access to these applications without compromising the application-specific performance and latency goals. In this paper, we empirically demonstrate that 5G radio communication becomes the side channel to precisely infer the user's applications in real-time. The key idea lies in observing the 5G physical and MAC layer interactions over time that reveal the application's behavior. The MAC layer receives the data from the application and requests the network to assign the radio resource blocks. The network assigns the radio resources as per application requirements, such as priority, Quality of Service (QoS) needs, amount of data to be transmitted, and buffer size. The adversary can passively observe the radio resources to fingerprint the applications. We empirically demonstrate this attack by considering four different categories of applications: online shopping, voice/video conferencing, video streaming, and Over-The-Top (OTT) media platforms. Finally, we have also demonstrated that an attacker can differentiate various types of applications in real-time within each category.



## **44. Adversarial Sparse Teacher: Defense Against Distillation-Based Model Stealing Attacks Using Adversarial Examples**

cs.LG

14 pages, 3 figures, 11 tables

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2403.05181v2) [paper-pdf](http://arxiv.org/pdf/2403.05181v2)

**Authors**: Eda Yilmaz, Hacer Yalim Keles

**Abstract**: We introduce Adversarial Sparse Teacher (AST), a robust defense method against distillation-based model stealing attacks. Our approach trains a teacher model using adversarial examples to produce sparse logit responses and increase the entropy of the output distribution. Typically, a model generates a peak in its output corresponding to its prediction. By leveraging adversarial examples, AST modifies the teacher model's original response, embedding a few altered logits into the output while keeping the primary response slightly higher. Concurrently, all remaining logits are elevated to further increase the output distribution's entropy. All these complex manipulations are performed using an optimization function with our proposed Exponential Predictive Divergence (EPD) loss function. EPD allows us to maintain higher entropy levels compared to traditional KL divergence, effectively confusing attackers. Experiments on CIFAR-10 and CIFAR-100 datasets demonstrate that AST outperforms state-of-the-art methods, providing effective defense against model stealing while preserving high accuracy. The source codes will be made publicly available here soon.



## **45. Flatness-aware Sequential Learning Generates Resilient Backdoors**

cs.LG

ECCV 2024

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14738v1) [paper-pdf](http://arxiv.org/pdf/2407.14738v1)

**Authors**: Hoang Pham, The-Anh Ta, Anh Tran, Khoa D. Doan

**Abstract**: Recently, backdoor attacks have become an emerging threat to the security of machine learning models. From the adversary's perspective, the implanted backdoors should be resistant to defensive algorithms, but some recently proposed fine-tuning defenses can remove these backdoors with notable efficacy. This is mainly due to the catastrophic forgetting (CF) property of deep neural networks. This paper counters CF of backdoors by leveraging continual learning (CL) techniques. We begin by investigating the connectivity between a backdoored and fine-tuned model in the loss landscape. Our analysis confirms that fine-tuning defenses, especially the more advanced ones, can easily push a poisoned model out of the backdoor regions, making it forget all about the backdoors. Based on this finding, we re-formulate backdoor training through the lens of CL and propose a novel framework, named Sequential Backdoor Learning (SBL), that can generate resilient backdoors. This framework separates the backdoor poisoning process into two tasks: the first task learns a backdoored model, while the second task, based on the CL principles, moves it to a backdoored region resistant to fine-tuning. We additionally propose to seek flatter backdoor regions via a sharpness-aware minimizer in the framework, further strengthening the durability of the implanted backdoor. Finally, we demonstrate the effectiveness of our method through extensive empirical experiments on several benchmark datasets in the backdoor domain. The source code is available at https://github.com/mail-research/SBL-resilient-backdoors



## **46. Bag of Tricks to Boost Adversarial Transferability**

cs.CV

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2401.08734v2) [paper-pdf](http://arxiv.org/pdf/2401.08734v2)

**Authors**: Zeliang Zhang, Wei Yao, Xiaosen Wang

**Abstract**: Deep neural networks are widely known to be vulnerable to adversarial examples. However, vanilla adversarial examples generated under the white-box setting often exhibit low transferability across different models. Since adversarial transferability poses more severe threats to practical applications, various approaches have been proposed for better transferability, including gradient-based, input transformation-based, and model-related attacks, \etc. In this work, we find that several tiny changes in the existing adversarial attacks can significantly affect the attack performance, \eg, the number of iterations and step size. Based on careful studies of existing adversarial attacks, we propose a bag of tricks to enhance adversarial transferability, including momentum initialization, scheduled step size, dual example, spectral-based input transformation, and several ensemble strategies. Extensive experiments on the ImageNet dataset validate the high effectiveness of our proposed tricks and show that combining them can further boost adversarial transferability. Our work provides practical insights and techniques to enhance adversarial transferability, and offers guidance to improve the attack performance on the real-world application through simple adjustments.



## **47. Augment then Smooth: Reconciling Differential Privacy with Certified Robustness**

cs.LG

29 pages, 19 figures. Accepted at TMLR in 2024. Link:  https://openreview.net/pdf?id=YN0IcnXqsr

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2306.08656v2) [paper-pdf](http://arxiv.org/pdf/2306.08656v2)

**Authors**: Jiapeng Wu, Atiyeh Ashari Ghomi, David Glukhov, Jesse C. Cresswell, Franziska Boenisch, Nicolas Papernot

**Abstract**: Machine learning models are susceptible to a variety of attacks that can erode trust, including attacks against the privacy of training data, and adversarial examples that jeopardize model accuracy. Differential privacy and certified robustness are effective frameworks for combating these two threats respectively, as they each provide future-proof guarantees. However, we show that standard differentially private model training is insufficient for providing strong certified robustness guarantees. Indeed, combining differential privacy and certified robustness in a single system is non-trivial, leading previous works to introduce complex training schemes that lack flexibility. In this work, we present DP-CERT, a simple and effective method that achieves both privacy and robustness guarantees simultaneously by integrating randomized smoothing into standard differentially private model training. Compared to the leading prior work, DP-CERT gives up to a 2.5% increase in certified accuracy for the same differential privacy guarantee on CIFAR10. Through in-depth persample metric analysis, we find that larger certifiable radii correlate with smaller local Lipschitz constants, and show that DP-CERT effectively reduces Lipschitz constants compared to other differentially private training methods. The code is available at github.com/layer6ailabs/dp-cert.



## **48. Human-Interpretable Adversarial Prompt Attack on Large Language Models with Situational Context**

cs.CL

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.14644v1) [paper-pdf](http://arxiv.org/pdf/2407.14644v1)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on testing the vulnerabilities in Large Language Models (LLMs) using adversarial attacks has primarily focused on nonsensical prompt injections, which are easily detected upon manual or automated review (e.g., via byte entropy). However, the exploration of innocuous human-understandable malicious prompts augmented with adversarial injections remains limited. In this research, we explore converting a nonsensical suffix attack into a sensible prompt via a situation-driven contextual re-writing. This allows us to show suffix conversion without any gradients, using only LLMs to perform the attacks, and thus better understand the scope of possible risks. We combine an independent, meaningful adversarial insertion and situations derived from movies to check if this can trick an LLM. The situations are extracted from the IMDB dataset, and prompts are defined following a few-shot chain-of-thought prompting. Our approach demonstrates that a successful situation-driven attack can be executed on both open-source and proprietary LLMs. We find that across many LLMs, as few as 1 attempt produces an attack and that these attacks transfer between LLMs. The link to our code is available at \url{https://anonymous.4open.science/r/Situation-Driven-Adversarial-Attacks-7BB1/README.md}.



## **49. Multi-Attribute Vision Transformers are Efficient and Robust Learners**

cs.CV

Accepted at IEEE ICIP 2024. arXiv admin note: text overlap with  arXiv:2207.08677 by other authors

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2402.08070v2) [paper-pdf](http://arxiv.org/pdf/2402.08070v2)

**Authors**: Hanan Gani, Nada Saadi, Noor Hussein, Karthik Nandakumar

**Abstract**: Since their inception, Vision Transformers (ViTs) have emerged as a compelling alternative to Convolutional Neural Networks (CNNs) across a wide spectrum of tasks. ViTs exhibit notable characteristics, including global attention, resilience against occlusions, and adaptability to distribution shifts. One underexplored aspect of ViTs is their potential for multi-attribute learning, referring to their ability to simultaneously grasp multiple attribute-related tasks. In this paper, we delve into the multi-attribute learning capability of ViTs, presenting a straightforward yet effective strategy for training various attributes through a single ViT network as distinct tasks. We assess the resilience of multi-attribute ViTs against adversarial attacks and compare their performance against ViTs designed for single attributes. Moreover, we further evaluate the robustness of multi-attribute ViTs against a recent transformer based attack called Patch-Fool. Our empirical findings on the CelebA dataset provide validation for our assertion. Our code is available at https://github.com/hananshafi/MTL-ViT



## **50. SlowPerception: Physical-World Latency Attack against Visual Perception in Autonomous Driving**

cs.CV

This submission was made without all contributors' consent

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2406.05800v2) [paper-pdf](http://arxiv.org/pdf/2406.05800v2)

**Authors**: Chen Ma, Ningfei Wang, Zhengyu Zhao, Qi Alfred Chen, Chao Shen

**Abstract**: Autonomous Driving (AD) systems critically depend on visual perception for real-time object detection and multiple object tracking (MOT) to ensure safe driving. However, high latency in these visual perception components can lead to significant safety risks, such as vehicle collisions. While previous research has extensively explored latency attacks within the digital realm, translating these methods effectively to the physical world presents challenges. For instance, existing attacks rely on perturbations that are unrealistic or impractical for AD, such as adversarial perturbations affecting areas like the sky, or requiring large patches that obscure most of a camera's view, thus making them impossible to be conducted effectively in the real world.   In this paper, we introduce SlowPerception, the first physical-world latency attack against AD perception, via generating projector-based universal perturbations. SlowPerception strategically creates numerous phantom objects on various surfaces in the environment, significantly increasing the computational load of Non-Maximum Suppression (NMS) and MOT, thereby inducing substantial latency. Our SlowPerception achieves second-level latency in physical-world settings, with an average latency of 2.5 seconds across different AD perception systems, scenarios, and hardware configurations. This performance significantly outperforms existing state-of-the-art latency attacks. Additionally, we conduct AD system-level impact assessments, such as vehicle collisions, using industry-grade AD systems with production-grade AD simulators with a 97% average rate. We hope that our analyses can inspire further research in this critical domain, enhancing the robustness of AD systems against emerging vulnerabilities.



