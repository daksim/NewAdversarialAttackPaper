# Latest Large Language Model Attack Papers
**update at 2024-10-19 09:50:37**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment**

cs.CL

28 pages

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13785v1) [paper-pdf](http://arxiv.org/pdf/2410.13785v1)

**Authors**: Zekun Moore Wang, Shawn Wang, Kang Zhu, Jiaheng Liu, Ke Xu, Jie Fu, Wangchunshu Zhou, Wenhao Huang

**Abstract**: Alignment of large language models (LLMs) involves training models on preference-contrastive output pairs to adjust their responses according to human preferences. To obtain such contrastive pairs, traditional methods like RLHF and RLAIF rely on limited contrasting patterns, such as varying model variants or decoding temperatures. This singularity leads to two issues: (1) alignment is not comprehensive; and thereby (2) models are susceptible to jailbreaking attacks. To address these issues, we investigate how to construct more comprehensive and diversified contrasting patterns to enhance preference data (RQ1) and verify the impact of the diversification of contrasting patterns on model alignment (RQ2). For RQ1, we propose PopAlign, a framework that integrates diversified contrasting patterns across the prompt, model, and pipeline levels, introducing six contrasting strategies that do not require additional feedback labeling procedures. Regarding RQ2, we conduct thorough experiments demonstrating that PopAlign significantly outperforms existing methods, leading to more comprehensive alignment.



## **2. Persistent Pre-Training Poisoning of LLMs**

cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13722v1) [paper-pdf](http://arxiv.org/pdf/2410.13722v1)

**Authors**: Yiming Zhang, Javier Rando, Ivan Evtimov, Jianfeng Chi, Eric Michael Smith, Nicholas Carlini, Florian Tramèr, Daphne Ippolito

**Abstract**: Large language models are pre-trained on uncurated text datasets consisting of trillions of tokens scraped from the Web. Prior work has shown that: (1) web-scraped pre-training datasets can be practically poisoned by malicious actors; and (2) adversaries can compromise language models after poisoning fine-tuning datasets. Our work evaluates for the first time whether language models can also be compromised during pre-training, with a focus on the persistence of pre-training attacks after models are fine-tuned as helpful and harmless chatbots (i.e., after SFT and DPO). We pre-train a series of LLMs from scratch to measure the impact of a potential poisoning adversary under four different attack objectives (denial-of-service, belief manipulation, jailbreaking, and prompt stealing), and across a wide range of model sizes (from 600M to 7B). Our main result is that poisoning only 0.1% of a model's pre-training dataset is sufficient for three out of four attacks to measurably persist through post-training. Moreover, simple attacks like denial-of-service persist through post-training with a poisoning rate of only 0.001%.



## **3. Jailbreaking LLM-Controlled Robots**

cs.RO

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13691v1) [paper-pdf](http://arxiv.org/pdf/2410.13691v1)

**Authors**: Alexander Robey, Zachary Ravichandran, Vijay Kumar, Hamed Hassani, George J. Pappas

**Abstract**: The recent introduction of large language models (LLMs) has revolutionized the field of robotics by enabling contextual reasoning and intuitive human-robot interaction in domains as varied as manipulation, locomotion, and self-driving vehicles. When viewed as a stand-alone technology, LLMs are known to be vulnerable to jailbreaking attacks, wherein malicious prompters elicit harmful text by bypassing LLM safety guardrails. To assess the risks of deploying LLMs in robotics, in this paper, we introduce RoboPAIR, the first algorithm designed to jailbreak LLM-controlled robots. Unlike existing, textual attacks on LLM chatbots, RoboPAIR elicits harmful physical actions from LLM-controlled robots, a phenomenon we experimentally demonstrate in three scenarios: (i) a white-box setting, wherein the attacker has full access to the NVIDIA Dolphins self-driving LLM, (ii) a gray-box setting, wherein the attacker has partial access to a Clearpath Robotics Jackal UGV robot equipped with a GPT-4o planner, and (iii) a black-box setting, wherein the attacker has only query access to the GPT-3.5-integrated Unitree Robotics Go2 robot dog. In each scenario and across three new datasets of harmful robotic actions, we demonstrate that RoboPAIR, as well as several static baselines, finds jailbreaks quickly and effectively, often achieving 100% attack success rates. Our results reveal, for the first time, that the risks of jailbroken LLMs extend far beyond text generation, given the distinct possibility that jailbroken robots could cause physical damage in the real world. Indeed, our results on the Unitree Go2 represent the first successful jailbreak of a deployed commercial robotic system. Addressing this emerging vulnerability is critical for ensuring the safe deployment of LLMs in robotics. Additional media is available at: https://robopair.org



## **4. Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation**

cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.09760v2) [paper-pdf](http://arxiv.org/pdf/2410.09760v2)

**Authors**: Guozhi Liu, Weiwei Lin, Tiansheng Huang, Ruichao Mo, Qi Mu, Li Shen

**Abstract**: Harmful fine-tuning attack poses a serious threat to the online fine-tuning service. Vaccine, a recent alignment-stage defense, applies uniform perturbation to all layers of embedding to make the model robust to the simulated embedding drift. However, applying layer-wise uniform perturbation may lead to excess perturbations for some particular safety-irrelevant layers, resulting in defense performance degradation and unnecessary memory consumption. To address this limitation, we propose Targeted Vaccine (T-Vaccine), a memory-efficient safety alignment method that applies perturbation to only selected layers of the model. T-Vaccine follows two core steps: First, it uses gradient norm as a statistical metric to identify the safety-critical layers. Second, instead of applying uniform perturbation across all layers, T-Vaccine only applies perturbation to the safety-critical layers while keeping other layers frozen during training. Results show that T-Vaccine outperforms Vaccine in terms of both defense effectiveness and resource efficiency. Comparison with other defense baselines, e.g., RepNoise and TAR also demonstrate the superiority of T-Vaccine. Notably, T-Vaccine is the first defense that can address harmful fine-tuning issues for a 7B pre-trained models trained on consumer GPUs with limited memory (e.g., RTX 4090). Our code is available at https://github.com/Lslland/T-Vaccine.



## **5. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.02240v3) [paper-pdf](http://arxiv.org/pdf/2410.02240v3)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.



## **6. Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning**

cs.CL

16 pages, 5 figures

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13274v1) [paper-pdf](http://arxiv.org/pdf/2410.13274v1)

**Authors**: Minseok Choi, ChaeHun Park, Dohyun Lee, Jaegul Choo

**Abstract**: Large language models (LLMs) serve as giant information stores, often including personal or copyrighted data, and retraining them from scratch is not a viable option. This has led to the development of various fast, approximate unlearning techniques to selectively remove knowledge from LLMs. Prior research has largely focused on minimizing the probabilities of specific token sequences by reversing the language modeling objective. However, these methods still leave LLMs vulnerable to adversarial attacks that exploit indirect references. In this work, we examine the limitations of current unlearning techniques in effectively erasing a particular type of indirect prompt: multi-hop queries. Our findings reveal that existing methods fail to completely remove multi-hop knowledge when one of the intermediate hops is unlearned. To address this issue, we propose MUNCH, a simple uncertainty-based approach that breaks down multi-hop queries into subquestions and leverages the uncertainty of the unlearned model in final decision-making. Empirical results demonstrate the effectiveness of our framework, and MUNCH can be easily integrated with existing unlearning techniques, making it a flexible and useful solution for enhancing unlearning processes.



## **7. FRAG: Toward Federated Vector Database Management for Collaborative and Secure Retrieval-Augmented Generation**

cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13272v1) [paper-pdf](http://arxiv.org/pdf/2410.13272v1)

**Authors**: Dongfang Zhao

**Abstract**: This paper introduces \textit{Federated Retrieval-Augmented Generation (FRAG)}, a novel database management paradigm tailored for the growing needs of retrieval-augmented generation (RAG) systems, which are increasingly powered by large-language models (LLMs). FRAG enables mutually-distrusted parties to collaboratively perform Approximate $k$-Nearest Neighbor (ANN) searches on encrypted query vectors and encrypted data stored in distributed vector databases, all while ensuring that no party can gain any knowledge about the queries or data of others. Achieving this paradigm presents two key challenges: (i) ensuring strong security guarantees, such as Indistinguishability under Chosen-Plaintext Attack (IND-CPA), under practical assumptions (e.g., we avoid overly optimistic assumptions like non-collusion among parties); and (ii) maintaining performance overheads comparable to traditional, non-federated RAG systems. To address these challenges, FRAG employs a single-key homomorphic encryption protocol that simplifies key management across mutually-distrusted parties. Additionally, FRAG introduces a \textit{multiplicative caching} technique to efficiently encrypt floating-point numbers, significantly improving computational performance in large-scale federated environments. We provide a rigorous security proof using standard cryptographic reductions and demonstrate the practical scalability and efficiency of FRAG through extensive experiments on both benchmark and real-world datasets.



## **8. Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis**

cs.CL

17 pages, 6 figures, 14 tables

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13237v1) [paper-pdf](http://arxiv.org/pdf/2410.13237v1)

**Authors**: Yiyi Chen, Qiongxiu Li, Russa Biswas, Johannes Bjerva

**Abstract**: Language Confusion is a phenomenon where Large Language Models (LLMs) generate text that is neither in the desired language, nor in a contextually appropriate language. This phenomenon presents a critical challenge in text generation by LLMs, often appearing as erratic and unpredictable behavior. We hypothesize that there are linguistic regularities to this inherent vulnerability in LLMs and shed light on patterns of language confusion across LLMs. We introduce a novel metric, Language Confusion Entropy, designed to directly measure and quantify this confusion, based on language distributions informed by linguistic typology and lexical variation. Comprehensive comparisons with the Language Confusion Benchmark (Marchisio et al., 2024) confirm the effectiveness of our metric, revealing patterns of language confusion across LLMs. We further link language confusion to LLM security, and find patterns in the case of multilingual embedding inversion attacks. Our analysis demonstrates that linguistic typology offers theoretically grounded interpretation, and valuable insights into leveraging language similarities as a prior for LLM alignment and security.



## **9. SPIN: Self-Supervised Prompt INjection**

cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13236v1) [paper-pdf](http://arxiv.org/pdf/2410.13236v1)

**Authors**: Leon Zhou, Junfeng Yang, Chengzhi Mao

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of important applications, yet their safety and reliability remain as major concerns. Various adversarial and jailbreak attacks have been proposed to bypass the safety alignment and cause the model to produce harmful responses. We introduce Self-supervised Prompt INjection (SPIN) which can detect and reverse these various attacks on LLMs. As our self-supervised prompt defense is done at inference-time, it is also compatible with existing alignment and adds an additional layer of safety for defense. Our benchmarks demonstrate that our system can reduce the attack success rate by up to 87.9%, while maintaining the performance on benign user requests. In addition, we discuss the situation of an adaptive attacker and show that our method is still resilient against attackers who are aware of our defense.



## **10. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

cs.CL

12 pages, 9 figures, EMNLP 2024 Findings

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2407.21659v4) [paper-pdf](http://arxiv.org/pdf/2407.21659v4)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.



## **11. SurrogatePrompt: Bypassing the Safety Filter of Text-to-Image Models via Substitution**

cs.CV

To appear in the the 31st ACM Conference on Computer and  Communications Security (CCS)

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2309.14122v3) [paper-pdf](http://arxiv.org/pdf/2309.14122v3)

**Authors**: Zhongjie Ba, Jieming Zhong, Jiachen Lei, Peng Cheng, Qinglong Wang, Zhan Qin, Zhibo Wang, Kui Ren

**Abstract**: Advanced text-to-image models such as DALL$\cdot$E 2 and Midjourney possess the capacity to generate highly realistic images, raising significant concerns regarding the potential proliferation of unsafe content. This includes adult, violent, or deceptive imagery of political figures. Despite claims of rigorous safety mechanisms implemented in these models to restrict the generation of not-safe-for-work (NSFW) content, we successfully devise and exhibit the first prompt attacks on Midjourney, resulting in the production of abundant photorealistic NSFW images. We reveal the fundamental principles of such prompt attacks and suggest strategically substituting high-risk sections within a suspect prompt to evade closed-source safety measures. Our novel framework, SurrogatePrompt, systematically generates attack prompts, utilizing large language models, image-to-text, and image-to-image modules to automate attack prompt creation at scale. Evaluation results disclose an 88% success rate in bypassing Midjourney's proprietary safety filter with our attack prompts, leading to the generation of counterfeit images depicting political figures in violent scenarios. Both subjective and objective assessments validate that the images generated from our attack prompts present considerable safety hazards.



## **12. Data Defenses Against Large Language Models**

cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13138v1) [paper-pdf](http://arxiv.org/pdf/2410.13138v1)

**Authors**: William Agnew, Harry H. Jiang, Cella Sum, Maarten Sap, Sauvik Das

**Abstract**: Large language models excel at performing inference over text to extract information, summarize information, or generate additional text. These inference capabilities are implicated in a variety of ethical harms spanning surveillance, labor displacement, and IP/copyright theft. While many policy, legal, and technical mitigations have been proposed to counteract these harms, these mitigations typically require cooperation from institutions that move slower than technical advances (i.e., governments) or that have few incentives to act to counteract these harms (i.e., the corporations that create and profit from these LLMs). In this paper, we define and build "data defenses" -- a novel strategy that directly empowers data owners to block LLMs from performing inference on their data. We create data defenses by developing a method to automatically generate adversarial prompt injections that, when added to input text, significantly reduce the ability of LLMs to accurately infer personally identifying information about the subject of the input text or to use copyrighted text in inference. We examine the ethics of enabling such direct resistance to LLM inference, and argue that making data defenses that resist and subvert LLMs enables the realization of important values such as data ownership, data sovereignty, and democratic control over AI systems. We verify that our data defenses are cheap and fast to generate, work on the latest commercial and open-source LLMs, resistance to countermeasures, and are robust to several different attack settings. Finally, we consider the security implications of LLM data defenses and outline several future research directions in this area. Our code is available at https://github.com/wagnew3/LLMDataDefenses and a tool for using our defenses to protect text against LLM inference is at https://wagnew3.github.io/LLM-Data-Defenses/.



## **13. Self-Comparison for Dataset-Level Membership Inference in Large (Vision-)Language Models**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.13088v1) [paper-pdf](http://arxiv.org/pdf/2410.13088v1)

**Authors**: Jie Ren, Kangrui Chen, Chen Chen, Vikash Sehwag, Yue Xing, Jiliang Tang, Lingjuan Lyu

**Abstract**: Large Language Models (LLMs) and Vision-Language Models (VLMs) have made significant advancements in a wide range of natural language processing and vision-language tasks. Access to large web-scale datasets has been a key factor in their success. However, concerns have been raised about the unauthorized use of copyrighted materials and potential copyright infringement. Existing methods, such as sample-level Membership Inference Attacks (MIA) and distribution-based dataset inference, distinguish member data (data used for training) and non-member data by leveraging the common observation that models tend to memorize and show greater confidence in member data. Nevertheless, these methods face challenges when applied to LLMs and VLMs, such as the requirement for ground-truth member data or non-member data that shares the same distribution as the test data. In this paper, we propose a novel dataset-level membership inference method based on Self-Comparison. We find that a member prefix followed by a non-member suffix (paraphrased from a member suffix) can further trigger the model's memorization on training data. Instead of directly comparing member and non-member data, we introduce paraphrasing to the second half of the sequence and evaluate how the likelihood changes before and after paraphrasing. Unlike prior approaches, our method does not require access to ground-truth member data or non-member data in identical distribution, making it more practical. Extensive experiments demonstrate that our proposed method outperforms traditional MIA and dataset inference techniques across various datasets and models, including including public models, fine-tuned models, and API-based commercial models.



## **14. Hiding-in-Plain-Sight (HiPS) Attack on CLIP for Targetted Object Removal from Images**

cs.LG

Published in the 3rd Workshop on New Frontiers in Adversarial Machine  Learning at NeurIPS 2024. 10 pages, 7 figures, 3 tables

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.13010v1) [paper-pdf](http://arxiv.org/pdf/2410.13010v1)

**Authors**: Arka Daw, Megan Hong-Thanh Chung, Maria Mahbub, Amir Sadovnik

**Abstract**: Machine learning models are known to be vulnerable to adversarial attacks, but traditional attacks have mostly focused on single-modalities. With the rise of large multi-modal models (LMMs) like CLIP, which combine vision and language capabilities, new vulnerabilities have emerged. However, prior work in multimodal targeted attacks aim to completely change the model's output to what the adversary wants. In many realistic scenarios, an adversary might seek to make only subtle modifications to the output, so that the changes go unnoticed by downstream models or even by humans. We introduce Hiding-in-Plain-Sight (HiPS) attacks, a novel class of adversarial attacks that subtly modifies model predictions by selectively concealing target object(s), as if the target object was absent from the scene. We propose two HiPS attack variants, HiPS-cls and HiPS-cap, and demonstrate their effectiveness in transferring to downstream image captioning models, such as CLIP-Cap, for targeted object removal from image captions.



## **15. Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization**

cs.LG

20 pages, 19 figures, 7 tables

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12949v1) [paper-pdf](http://arxiv.org/pdf/2410.12949v1)

**Authors**: Phillip Guo, Aaquib Syed, Abhay Sheshadri, Aidan Ewart, Gintare Karolina Dziugaite

**Abstract**: Methods for knowledge editing and unlearning in large language models seek to edit or remove undesirable knowledge or capabilities without compromising general language modeling performance. This work investigates how mechanistic interpretability -- which, in part, aims to identify model components (circuits) associated to specific interpretable mechanisms that make up a model capability -- can improve the precision and effectiveness of editing and unlearning. We find a stark difference in unlearning and edit robustness when training components localized by different methods. We highlight an important distinction between methods that localize components based primarily on preserving outputs, and those finding high level mechanisms with predictable intermediate states. In particular, localizing edits/unlearning to components associated with the lookup-table mechanism for factual recall 1) leads to more robust edits/unlearning across different input/output formats, and 2) resists attempts to relearn the unwanted information, while also reducing unintended side effects compared to baselines, on both a sports facts dataset and the CounterFact dataset across multiple models. We also find that certain localized edits disrupt the latent knowledge in the model more than any other baselines, making unlearning more robust to various attacks.



## **16. ToBlend: Token-Level Blending With an Ensemble of LLMs to Attack AI-Generated Text Detection**

cs.CL

Submitted to ARR Oct-2024 Cycle

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2402.11167v2) [paper-pdf](http://arxiv.org/pdf/2402.11167v2)

**Authors**: Fan Huang, Haewoon Kwak, Jisun An

**Abstract**: The robustness of AI-content detection models against sophisticated adversarial strategies, such as paraphrasing or word switching, is a rising concern in natural language generation (NLG) applications. This study proposes ToBlend, a novel token-level ensemble text generation method to challenge the robustness of current AI-content detection approaches by utilizing multiple sets of candidate generative large language models (LLMs). By randomly sampling token(s) from candidate LLMs sets, we find ToBlend significantly drops the performance of most mainstream AI-content detection methods. We evaluate the text quality produced under different ToBlend settings based on annotations from experienced human experts. We proposed a fine-tuned Llama3.1 model to distinguish the ToBlend generated text more accurately. Our findings underscore our proposed text generation approach's great potential in deceiving and improving detection models. Our datasets, codes, and annotations are open-sourced.



## **17. Efficient and Effective Universal Adversarial Attack against Vision-Language Pre-training Models**

cs.CV

11 pages

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.11639v2) [paper-pdf](http://arxiv.org/pdf/2410.11639v2)

**Authors**: Fan Yang, Yihao Huang, Kailong Wang, Ling Shi, Geguang Pu, Yang Liu, Haoyu Wang

**Abstract**: Vision-language pre-training (VLP) models, trained on large-scale image-text pairs, have become widely used across a variety of downstream vision-and-language (V+L) tasks. This widespread adoption raises concerns about their vulnerability to adversarial attacks. Non-universal adversarial attacks, while effective, are often impractical for real-time online applications due to their high computational demands per data instance. Recently, universal adversarial perturbations (UAPs) have been introduced as a solution, but existing generator-based UAP methods are significantly time-consuming. To overcome the limitation, we propose a direct optimization-based UAP approach, termed DO-UAP, which significantly reduces resource consumption while maintaining high attack performance. Specifically, we explore the necessity of multimodal loss design and introduce a useful data augmentation strategy. Extensive experiments conducted on three benchmark VLP datasets, six popular VLP models, and three classical downstream tasks demonstrate the efficiency and effectiveness of DO-UAP. Specifically, our approach drastically decreases the time consumption by 23-fold while achieving a better attack performance.



## **18. Reconstruction of Differentially Private Text Sanitization via Large Language Models**

cs.CR

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12443v1) [paper-pdf](http://arxiv.org/pdf/2410.12443v1)

**Authors**: Shuchao Pang, Zhigang Lu, Haichen Wang, Peng Fu, Yongbin Zhou, Minhui Xue, Bo Li

**Abstract**: Differential privacy (DP) is the de facto privacy standard against privacy leakage attacks, including many recently discovered ones against large language models (LLMs). However, we discovered that LLMs could reconstruct the altered/removed privacy from given DP-sanitized prompts. We propose two attacks (black-box and white-box) based on the accessibility to LLMs and show that LLMs could connect the pair of DP-sanitized text and the corresponding private training data of LLMs by giving sample text pairs as instructions (in the black-box attacks) or fine-tuning data (in the white-box attacks). To illustrate our findings, we conduct comprehensive experiments on modern LLMs (e.g., LLaMA-2, LLaMA-3, ChatGPT-3.5, ChatGPT-4, ChatGPT-4o, Claude-3, Claude-3.5, OPT, GPT-Neo, GPT-J, Gemma-2, and Pythia) using commonly used datasets (such as WikiMIA, Pile-CC, and Pile-Wiki) against both word-level and sentence-level DP. The experimental results show promising recovery rates, e.g., the black-box attacks against the word-level DP over WikiMIA dataset gave 72.18% on LLaMA-2 (70B), 82.39% on LLaMA-3 (70B), 75.35% on Gemma-2, 91.2% on ChatGPT-4o, and 94.01% on Claude-3.5 (Sonnet). More urgently, this study indicates that these well-known LLMs have emerged as a new security risk for existing DP text sanitization approaches in the current environment.



## **19. Can LLMs Patch Security Issues?**

cs.CR

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2312.00024v5) [paper-pdf](http://arxiv.org/pdf/2312.00024v5)

**Authors**: Kamel Alrashedy, Abdullah Aljasser, Pradyumna Tambwekar, Matthew Gombolay

**Abstract**: Large Language Models (LLMs) have shown impressive proficiency in code generation. Unfortunately, these models share a weakness with their human counterparts: producing code that inadvertently has security vulnerabilities. These vulnerabilities could allow unauthorized attackers to access sensitive data or systems, which is unacceptable for safety-critical applications. In this work, we propose Feedback-Driven Security Patching (FDSP), where LLMs automatically refine generated, vulnerable code. Our approach leverages automatic static code analysis to empower the LLM to generate and implement potential solutions to address vulnerabilities. We address the research communitys needs for safe code generation by introducing a large-scale dataset, PythonSecurityEval, covering the diversity of real-world applications, including databases, websites and operating systems. We empirically validate that FDSP outperforms prior work that uses self-feedback from LLMs by up to 17.6% through our procedure that injects targeted, external feedback. Code and data are available at \url{https://github.com/Kamel773/LLM-code-refine}



## **20. GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation**

cs.CR

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.13077v2) [paper-pdf](http://arxiv.org/pdf/2405.13077v2)

**Authors**: Govind Ramesh, Yao Dou, Wei Xu

**Abstract**: Research on jailbreaking has been valuable for testing and understanding the safety and security issues of large language models (LLMs). In this paper, we introduce Iterative Refinement Induced Self-Jailbreak (IRIS), a novel approach that leverages the reflective capabilities of LLMs for jailbreaking with only black-box access. Unlike previous methods, IRIS simplifies the jailbreaking process by using a single model as both the attacker and target. This method first iteratively refines adversarial prompts through self-explanation, which is crucial for ensuring that even well-aligned LLMs obey adversarial instructions. IRIS then rates and enhances the output given the refined prompt to increase its harmfulness. We find that IRIS achieves jailbreak success rates of 98% on GPT-4, 92% on GPT-4 Turbo, and 94% on Llama-3.1-70B in under 7 queries. It significantly outperforms prior approaches in automatic, black-box, and interpretable jailbreaking, while requiring substantially fewer queries, thereby establishing a new standard for interpretable jailbreaking methods.



## **21. Taking off the Rose-Tinted Glasses: A Critical Look at Adversarial ML Through the Lens of Evasion Attacks**

cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.12076v1) [paper-pdf](http://arxiv.org/pdf/2410.12076v1)

**Authors**: Kevin Eykholt, Farhan Ahmed, Pratik Vaishnavi, Amir Rahmati

**Abstract**: The vulnerability of machine learning models in adversarial scenarios has garnered significant interest in the academic community over the past decade, resulting in a myriad of attacks and defenses. However, while the community appears to be overtly successful in devising new attacks across new contexts, the development of defenses has stalled. After a decade of research, we appear no closer to securing AI applications beyond additional training. Despite a lack of effective mitigations, AI development and its incorporation into existing systems charge full speed ahead with the rise of generative AI and large language models. Will our ineffectiveness in developing solutions to adversarial threats further extend to these new technologies?   In this paper, we argue that overly permissive attack and overly restrictive defensive threat models have hampered defense development in the ML domain. Through the lens of adversarial evasion attacks against neural networks, we critically examine common attack assumptions, such as the ability to bypass any defense not explicitly built into the model. We argue that these flawed assumptions, seen as reasonable by the community based on paper acceptance, have encouraged the development of adversarial attacks that map poorly to real-world scenarios. In turn, new defenses evaluated against these very attacks are inadvertently required to be almost perfect and incorporated as part of the model. But do they need to? In practice, machine learning models are deployed as a small component of a larger system. We analyze adversarial machine learning from a system security perspective rather than an AI perspective and its implications for emerging AI paradigms.



## **22. A Watermark for Low-entropy and Unbiased Generation in Large Language Models**

cs.CL

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.14604v2) [paper-pdf](http://arxiv.org/pdf/2405.14604v2)

**Authors**: Minjia Mao, Dongjun Wei, Zeyu Chen, Xiao Fang, Michael Chau

**Abstract**: Recent advancements in large language models (LLMs) have highlighted the risk of misusing them, raising the need for accurate detection of LLM-generated content. In response, a viable solution is to inject imperceptible identifiers into LLMs, known as watermarks. Previous work demonstrates that unbiased watermarks ensure unforgeability and preserve text quality by maintaining the expectation of the LLM output probability distribution. However, previous unbiased watermarking methods suffer from one or more of the following issues: (1) requiring access to white-box LLMs during detection, (2) incurring long detection time, (3) being not robust against simple watermarking attacks, (4) failing to provide statistical guarantees for the type II error of watermark detection, and (5) being not statistically unbiased for low-entropy scenarios, which hinder their deployment in practice. This study proposes the Sampling One Then Accepting (STA-1) method, a watermark that can address all of these issues. Moreover, we discuss the tradeoff between watermark strength and text quality for unbiased watermarks. We show that in low-entropy scenarios, unbiased watermarks face a tradeoff between watermark strength and the risk of unsatisfactory outputs. Experimental results on both low-entropy and high-entropy datasets demonstrate that STA-1 achieves text quality and watermark strength comparable to existing unbiased watermarks, with a low risk of unsatisfactory outputs. Implementation codes for this study are available online.



## **23. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

cs.MA

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11782v1) [paper-pdf](http://arxiv.org/pdf/2410.11782v1)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.



## **24. LLM-Based Robust Product Classification in Commerce and Compliance**

cs.CL

Camera-ready version for Customizable NLP Workshop at EMNLP 2024. 11  pages

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2408.05874v2) [paper-pdf](http://arxiv.org/pdf/2408.05874v2)

**Authors**: Sina Gholamian, Gianfranco Romani, Bartosz Rudnikowicz, Stavroula Skylaki

**Abstract**: Product classification is a crucial task in international trade, as compliance regulations are verified and taxes and duties are applied based on product categories. Manual classification of products is time-consuming and error-prone, and the sheer volume of products imported and exported renders the manual process infeasible. Consequently, e-commerce platforms and enterprises involved in international trade have turned to automatic product classification using machine learning. However, current approaches do not consider the real-world challenges associated with product classification, such as very abbreviated and incomplete product descriptions. In addition, recent advancements in generative Large Language Models (LLMs) and their reasoning capabilities are mainly untapped in product classification and e-commerce. In this research, we explore the real-life challenges of industrial classification and we propose data perturbations that allow for realistic data simulation. Furthermore, we employ LLM-based product classification to improve the robustness of the prediction in presence of incomplete data. Our research shows that LLMs with in-context learning outperform the supervised approaches in the clean-data scenario. Additionally, we illustrate that LLMs are significantly more robust than the supervised approaches when data attacks are present.



## **25. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.20485v2) [paper-pdf](http://arxiv.org/pdf/2405.20485v2)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs), by anchoring, adapting, and personalizing their responses to the most relevant knowledge sources. It is particularly useful in chatbot applications, allowing developers to customize LLM output without expensive retraining. Despite their significant utility in various applications, RAG systems present new security risks. In this work, we propose new attack vectors that allow an adversary to inject a single malicious document into a RAG system's knowledge base, and mount a backdoor poisoning attack. We design Phantom, a general two-stage optimization framework against RAG systems, that crafts a malicious poisoned document leading to an integrity violation in the model's output. First, the document is constructed to be retrieved only when a specific trigger sequence of tokens appears in the victim's queries. Second, the document is further optimized with crafted adversarial text that induces various adversarial objectives on the LLM output, including refusal to answer, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama, and show that they transfer to GPT-3.5 Turbo and GPT-4. Finally, we successfully conducted a Phantom attack on NVIDIA's black-box production RAG system, "Chat with RTX".



## **26. Gotcha! This Model Uses My Code! Evaluating Membership Leakage Risks in Code Models**

cs.SE

Accepted by IEEE Transactions on Software Engineering, Camera-Ready  Version

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2310.01166v2) [paper-pdf](http://arxiv.org/pdf/2310.01166v2)

**Authors**: Zhou Yang, Zhipeng Zhao, Chenyu Wang, Jieke Shi, Dongsum Kim, Donggyun Han, David Lo

**Abstract**: Given large-scale source code datasets available in open-source projects and advanced large language models, recent code models have been proposed to address a series of critical software engineering tasks, such as program repair and code completion. The training data of the code models come from various sources, not only the publicly available source code, e.g., open-source projects on GitHub but also the private data such as the confidential source code from companies, which may contain sensitive information (for example, SSH keys and personal information). As a result, the use of these code models may raise new privacy concerns.   In this paper, we focus on a critical yet not well-explored question on using code models: what is the risk of membership information leakage in code models? Membership information leakage refers to the risk that an attacker can infer whether a given data point is included in (i.e., a member of) the training data. To answer this question, we propose Gotcha, a novel membership inference attack method specifically for code models. We investigate the membership leakage risk of code models. Our results reveal a worrying fact that the risk of membership leakage is high: although the previous attack methods are close to random guessing, Gotcha can predict the data membership with a high true positive rate of 0.95 and a low false positive rate of 0.10. We also show that the attacker's knowledge of the victim model (e.g., the model architecture and the pre-training data) impacts the success rate of attacks. Further analysis demonstrates that changing the decoding strategy can mitigate the risk of membership leakage. This study calls for more attention to understanding the privacy of code models and developing more effective countermeasures against such attacks.



## **27. Multi-round jailbreak attack on large language models**

cs.CL

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11533v1) [paper-pdf](http://arxiv.org/pdf/2410.11533v1)

**Authors**: Yihua Zhou, Xiaochuan Shi

**Abstract**: Ensuring the safety and alignment of large language models (LLMs) with human values is crucial for generating responses that are beneficial to humanity. While LLMs have the capability to identify and avoid harmful queries, they remain vulnerable to "jailbreak" attacks, where carefully crafted prompts can induce the generation of toxic content. Traditional single-round jailbreak attacks, such as GCG and AutoDAN, do not alter the sensitive words in the dangerous prompts. Although they can temporarily bypass the model's safeguards through prompt engineering, their success rate drops significantly as the LLM is further fine-tuned, and they cannot effectively circumvent static rule-based filters that remove the hazardous vocabulary.   In this study, to better understand jailbreak attacks, we introduce a multi-round jailbreak approach. This method can rewrite the dangerous prompts, decomposing them into a series of less harmful sub-questions to bypass the LLM's safety checks. We first use the LLM to perform a decomposition task, breaking down a set of natural language questions into a sequence of progressive sub-questions, which are then used to fine-tune the Llama3-8B model, enabling it to decompose hazardous prompts. The fine-tuned model is then used to break down the problematic prompt, and the resulting sub-questions are sequentially asked to the victim model. If the victim model rejects a sub-question, a new decomposition is generated, and the process is repeated until the final objective is achieved. Our experimental results show a 94\% success rate on the llama2-7B and demonstrate the effectiveness of this approach in circumventing static rule-based filters.



## **28. Jigsaw Puzzles: Splitting Harmful Questions to Jailbreak Large Language Models**

cs.CL

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11459v1) [paper-pdf](http://arxiv.org/pdf/2410.11459v1)

**Authors**: Hao Yang, Lizhen Qu, Ehsan Shareghi, Gholamreza Haffari

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in engaging with humans and addressing complex questions by leveraging their vast implicit knowledge and robust reasoning capabilities. However, such models are vulnerable to jailbreak attacks, leading to the generation of harmful responses. Despite recent research on single-turn jailbreak strategies to facilitate the development of defence mechanisms, the challenge of revealing vulnerabilities under multi-turn setting remains relatively under-explored. In this work, we propose Jigsaw Puzzles (JSP), a straightforward yet effective multi-turn jailbreak strategy against the advanced LLMs. JSP splits questions into harmless fractions as the input of each turn, and requests LLMs to reconstruct and respond to questions under multi-turn interaction. Our experimental results demonstrate that the proposed JSP jailbreak bypasses original safeguards against explicitly harmful content, achieving an average attack success rate of 93.76% on 189 harmful queries across 5 advanced LLMs (Gemini-1.5-Pro, Llama-3.1-70B, GPT-4, GPT-4o, GPT-4o-mini). Moreover, JSP achieves a state-of-the-art attack success rate of 92% on GPT-4 on the harmful query benchmark, and exhibits strong resistant to defence strategies. Warning: this paper contains offensive examples.



## **29. Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation**

cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11317v1) [paper-pdf](http://arxiv.org/pdf/2410.11317v1)

**Authors**: Qizhang Li, Xiaochen Yang, Wangmeng Zuo, Yiwen Guo

**Abstract**: Automatic adversarial prompt generation provides remarkable success in jailbreaking safely-aligned large language models (LLMs). Existing gradient-based attacks, while demonstrating outstanding performance in jailbreaking white-box LLMs, often generate garbled adversarial prompts with chaotic appearance. These adversarial prompts are difficult to transfer to other LLMs, hindering their performance in attacking unknown victim models. In this paper, for the first time, we delve into the semantic meaning embedded in garbled adversarial prompts and propose a novel method that "translates" them into coherent and human-readable natural language adversarial prompts. In this way, we can effectively uncover the semantic information that triggers vulnerabilities of the model and unambiguously transfer it to the victim model, without overlooking the adversarial information hidden in the garbled text, to enhance jailbreak attacks. It also offers a new approach to discovering effective designs for jailbreak prompts, advancing the understanding of jailbreak attacks. Experimental results demonstrate that our method significantly improves the success rate of jailbreak attacks against various safety-aligned LLMs and outperforms state-of-the-arts by large margins. With at most 10 queries, our method achieves an average attack success rate of 81.8% in attacking 7 commercial closed-source LLMs, including GPT and Claude-3 series, on HarmBench. Our method also achieves over 90% attack success rates against Llama-2-Chat models on AdvBench, despite their outstanding resistance to jailbreak attacks. Code at: https://github.com/qizhangli/Adversarial-Prompt-Translator.



## **30. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation**

cs.CV

ECCV2024 (Project Page: https://gyhdog99.github.io/projects/ecso/)

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2403.09572v4) [paper-pdf](http://arxiv.org/pdf/2403.09572v4)

**Authors**: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang

**Abstract**: Multimodal large language models (MLLMs) have shown impressive reasoning abilities. However, they are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting the unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed with the introduction of image features. To construct robust MLLMs, we propose ECSO (Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate the intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that ECSO enhances model safety significantly (e.g.,, 37.6% improvement on the MM-SafetyBench (SD+OCR) and 71.3% on VLSafe with LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.



## **31. Cognitive Overload Attack:Prompt Injection for Long Context**

cs.CL

40 pages, 31 Figures

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11272v1) [paper-pdf](http://arxiv.org/pdf/2410.11272v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Amin Karbasi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in performing tasks across various domains without needing explicit retraining. This capability, known as In-Context Learning (ICL), while impressive, exposes LLMs to a variety of adversarial prompts and jailbreaks that manipulate safety-trained LLMs into generating undesired or harmful output. In this paper, we propose a novel interpretation of ICL in LLMs through the lens of cognitive neuroscience, by drawing parallels between learning in human cognition with ICL. We applied the principles of Cognitive Load Theory in LLMs and empirically validate that similar to human cognition, LLMs also suffer from cognitive overload a state where the demand on cognitive processing exceeds the available capacity of the model, leading to potential errors. Furthermore, we demonstrated how an attacker can exploit ICL to jailbreak LLMs through deliberately designed prompts that induce cognitive overload on LLMs, thereby compromising the safety mechanisms of LLMs. We empirically validate this threat model by crafting various cognitive overload prompts and show that advanced models such as GPT-4, Claude-3.5 Sonnet, Claude-3 OPUS, Llama-3-70B-Instruct, Gemini-1.0-Pro, and Gemini-1.5-Pro can be successfully jailbroken, with attack success rates of up to 99.99%. Our findings highlight critical vulnerabilities in LLMs and underscore the urgency of developing robust safeguards. We propose integrating insights from cognitive load theory into the design and evaluation of LLMs to better anticipate and mitigate the risks of adversarial attacks. By expanding our experiments to encompass a broader range of models and by highlighting vulnerabilities in LLMs' ICL, we aim to ensure the development of safer and more reliable AI systems.



## **32. Archilles' Heel in Semi-open LLMs: Hiding Bottom against Recovery Attacks**

cs.LG

10 pages for main content of the paper

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11182v1) [paper-pdf](http://arxiv.org/pdf/2410.11182v1)

**Authors**: Hanbo Huang, Yihan Li, Bowen Jiang, Lin Liu, Ruoyu Sun, Zhuotao Liu, Shiyu Liang

**Abstract**: Closed-source large language models deliver strong performance but have limited downstream customizability. Semi-open models, combining both closed-source and public layers, were introduced to improve customizability. However, parameters in the closed-source layers are found vulnerable to recovery attacks. In this paper, we explore the design of semi-open models with fewer closed-source layers, aiming to increase customizability while ensuring resilience to recovery attacks. We analyze the contribution of closed-source layer to the overall resilience and theoretically prove that in a deep transformer-based model, there exists a transition layer such that even small recovery errors in layers before this layer can lead to recovery failure. Building on this, we propose \textbf{SCARA}, a novel approach that keeps only a few bottom layers as closed-source. SCARA employs a fine-tuning-free metric to estimate the maximum number of layers that can be publicly accessible for customization. We apply it to five models (1.3B to 70B parameters) to construct semi-open models, validating their customizability on six downstream tasks and assessing their resilience against various recovery attacks on sixteen benchmarks. We compare SCARA to baselines and observe that it generally improves downstream customization performance and offers similar resilience with over \textbf{10} times fewer closed-source parameters. We empirically investigate the existence of transition layers, analyze the effectiveness of our scheme and finally discuss its limitations.



## **33. Denial-of-Service Poisoning Attacks against Large Language Models**

cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10760v1) [paper-pdf](http://arxiv.org/pdf/2410.10760v1)

**Authors**: Kuofeng Gao, Tianyu Pang, Chao Du, Yong Yang, Shu-Tao Xia, Min Lin

**Abstract**: Recent studies have shown that LLMs are vulnerable to denial-of-service (DoS) attacks, where adversarial inputs like spelling errors or non-semantic prompts trigger endless outputs without generating an [EOS] token. These attacks can potentially cause high latency and make LLM services inaccessible to other users or tasks. However, when there are speech-to-text interfaces (e.g., voice commands to a robot), executing such DoS attacks becomes challenging, as it is difficult to introduce spelling errors or non-semantic prompts through speech. A simple DoS attack in these scenarios would be to instruct the model to "Keep repeating Hello", but we observe that relying solely on natural instructions limits output length, which is bounded by the maximum length of the LLM's supervised finetuning (SFT) data. To overcome this limitation, we propose poisoning-based DoS (P-DoS) attacks for LLMs, demonstrating that injecting a single poisoned sample designed for DoS purposes can break the output length limit. For example, a poisoned sample can successfully attack GPT-4o and GPT-4o mini (via OpenAI's finetuning API) using less than $1, causing repeated outputs up to the maximum inference length (16K tokens, compared to 0.5K before poisoning). Additionally, we perform comprehensive ablation studies on open-source LLMs and extend our method to LLM agents, where attackers can control both the finetuning dataset and algorithm. Our findings underscore the urgent need for defenses against P-DoS attacks to secure LLMs. Our code is available at https://github.com/sail-sg/P-DoS.



## **34. Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues**

cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10700v1) [paper-pdf](http://arxiv.org/pdf/2410.10700v1)

**Authors**: Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, Jing Shao

**Abstract**: This study exposes the safety vulnerabilities of Large Language Models (LLMs) in multi-turn interactions, where malicious users can obscure harmful intents across several queries. We introduce ActorAttack, a novel multi-turn attack method inspired by actor-network theory, which models a network of semantically linked actors as attack clues to generate diverse and effective attack paths toward harmful targets. ActorAttack addresses two main challenges in multi-turn attacks: (1) concealing harmful intents by creating an innocuous conversation topic about the actor, and (2) uncovering diverse attack paths towards the same harmful target by leveraging LLMs' knowledge to specify the correlated actors as various attack clues. In this way, ActorAttack outperforms existing single-turn and multi-turn attack methods across advanced aligned LLMs, even for GPT-o1. We will publish a dataset called SafeMTData, which includes multi-turn adversarial prompts and safety alignment data, generated by ActorAttack. We demonstrate that models safety-tuned using our safety dataset are more robust to multi-turn attacks. Code is available at https://github.com/renqibing/ActorAttack.



## **35. F2A: An Innovative Approach for Prompt Injection by Utilizing Feign Security Detection Agents**

cs.CR

1. Fixed typo in abstract 2. Provisionally completed the article  update to facilitate future version revisions

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.08776v2) [paper-pdf](http://arxiv.org/pdf/2410.08776v2)

**Authors**: Yupeng Ren

**Abstract**: With the rapid development of Large Language Models (LLMs), numerous mature applications of LLMs have emerged in the field of content safety detection. However, we have found that LLMs exhibit blind trust in safety detection agents. The general LLMs can be compromised by hackers with this vulnerability. Hence, this paper proposed an attack named Feign Agent Attack (F2A).Through such malicious forgery methods, adding fake safety detection results into the prompt, the defense mechanism of LLMs can be bypassed, thereby obtaining harmful content and hijacking the normal conversation. Continually, a series of experiments were conducted. In these experiments, the hijacking capability of F2A on LLMs was analyzed and demonstrated, exploring the fundamental reasons why LLMs blindly trust safety detection results. The experiments involved various scenarios where fake safety detection results were injected into prompts, and the responses were closely monitored to understand the extent of the vulnerability. Also, this paper provided a reasonable solution to this attack, emphasizing that it is important for LLMs to critically evaluate the results of augmented agents to prevent the generating harmful content. By doing so, the reliability and security can be significantly improved, protecting the LLMs from F2A.



## **36. On Calibration of LLM-based Guard Models for Reliable Content Moderation**

cs.CR

19 pages, 9 figures

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10414v1) [paper-pdf](http://arxiv.org/pdf/2410.10414v1)

**Authors**: Hongfu Liu, Hengguan Huang, Hao Wang, Xiangming Gu, Ye Wang

**Abstract**: Large language models (LLMs) pose significant risks due to the potential for generating harmful content or users attempting to evade guardrails. Existing studies have developed LLM-based guard models designed to moderate the input and output of threat LLMs, ensuring adherence to safety policies by blocking content that violates these protocols upon deployment. However, limited attention has been given to the reliability and calibration of such guard models. In this work, we empirically conduct comprehensive investigations of confidence calibration for 9 existing LLM-based guard models on 12 benchmarks in both user input and model output classification. Our findings reveal that current LLM-based guard models tend to 1) produce overconfident predictions, 2) exhibit significant miscalibration when subjected to jailbreak attacks, and 3) demonstrate limited robustness to the outputs generated by different types of response models. Additionally, we assess the effectiveness of post-hoc calibration methods to mitigate miscalibration. We demonstrate the efficacy of temperature scaling and, for the first time, highlight the benefits of contextual calibration for confidence calibration of guard models, particularly in the absence of validation sets. Our analysis and experiments underscore the limitations of current LLM-based guard models and provide valuable insights for the future development of well-calibrated guard models toward more reliable content moderation. We also advocate for incorporating reliability evaluation of confidence calibration when releasing future LLM-based guard models.



## **37. Jailbreak Instruction-Tuned LLMs via end-of-sentence MLP Re-weighting**

cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10150v1) [paper-pdf](http://arxiv.org/pdf/2410.10150v1)

**Authors**: Yifan Luo, Zhennan Zhou, Meitan Wang, Bin Dong

**Abstract**: In this paper, we investigate the safety mechanisms of instruction fine-tuned large language models (LLMs). We discover that re-weighting MLP neurons can significantly compromise a model's safety, especially for MLPs in end-of-sentence inferences. We hypothesize that LLMs evaluate the harmfulness of prompts during end-of-sentence inferences, and MLP layers plays a critical role in this process. Based on this hypothesis, we develop 2 novel white-box jailbreak methods: a prompt-specific method and a prompt-general method. The prompt-specific method targets individual prompts and optimizes the attack on the fly, while the prompt-general method is pre-trained offline and can generalize to unseen harmful prompts. Our methods demonstrate robust performance across 7 popular open-source LLMs, size ranging from 2B to 72B. Furthermore, our study provides insights into vulnerabilities of instruction-tuned LLM's safety and deepens the understanding of the internal mechanisms of LLMs.



## **38. White-box Multimodal Jailbreaks Against Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2405.17894v2) [paper-pdf](http://arxiv.org/pdf/2405.17894v2)

**Authors**: Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji, Guangnan Ye, Yu-Gang Jiang

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have underscored their superiority in various multimodal tasks. However, the adversarial robustness of VLMs has not been fully explored. Existing methods mainly assess robustness through unimodal adversarial attacks that perturb images, while assuming inherent resilience against text-based attacks. Different from existing attacks, in this work we propose a more comprehensive strategy that jointly attacks both text and image modalities to exploit a broader spectrum of vulnerability within VLMs. Specifically, we propose a dual optimization objective aimed at guiding the model to generate affirmative responses with high toxicity. Our attack method begins by optimizing an adversarial image prefix from random noise to generate diverse harmful responses in the absence of text input, thus imbuing the image with toxic semantics. Subsequently, an adversarial text suffix is integrated and co-optimized with the adversarial image prefix to maximize the probability of eliciting affirmative responses to various harmful instructions. The discovered adversarial image prefix and text suffix are collectively denoted as a Universal Master Key (UMK). When integrated into various malicious queries, UMK can circumvent the alignment defenses of VLMs and lead to the generation of objectionable content, known as jailbreaks. The experimental results demonstrate that our universal attack strategy can effectively jailbreak MiniGPT-4 with a 96% success rate, highlighting the vulnerability of VLMs and the urgent need for new alignment strategies.



## **39. BlackDAN: A Black-Box Multi-Objective Approach for Effective and Contextual Jailbreaking of Large Language Models**

cs.CR

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09804v1) [paper-pdf](http://arxiv.org/pdf/2410.09804v1)

**Authors**: Xinyuan Wang, Victor Shea-Jay Huang, Renmiao Chen, Hao Wang, Chengwei Pan, Lei Sha, Minlie Huang

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across various tasks, they encounter potential security risks such as jailbreak attacks, which exploit vulnerabilities to bypass security measures and generate harmful outputs. Existing jailbreak strategies mainly focus on maximizing attack success rate (ASR), frequently neglecting other critical factors, including the relevance of the jailbreak response to the query and the level of stealthiness. This narrow focus on single objectives can result in ineffective attacks that either lack contextual relevance or are easily recognizable. In this work, we introduce BlackDAN, an innovative black-box attack framework with multi-objective optimization, aiming to generate high-quality prompts that effectively facilitate jailbreaking while maintaining contextual relevance and minimizing detectability. BlackDAN leverages Multiobjective Evolutionary Algorithms (MOEAs), specifically the NSGA-II algorithm, to optimize jailbreaks across multiple objectives including ASR, stealthiness, and semantic relevance. By integrating mechanisms like mutation, crossover, and Pareto-dominance, BlackDAN provides a transparent and interpretable process for generating jailbreaks. Furthermore, the framework allows customization based on user preferences, enabling the selection of prompts that balance harmfulness, relevance, and other factors. Experimental results demonstrate that BlackDAN outperforms traditional single-objective methods, yielding higher success rates and improved robustness across various LLMs and multimodal LLMs, while ensuring jailbreak responses are both relevant and less detectable.



## **40. 'Quis custodiet ipsos custodes?' Who will watch the watchmen? On Detecting AI-generated peer-reviews**

cs.CL

EMNLP Main, 17 pages, 5 figures, 9 tables

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09770v1) [paper-pdf](http://arxiv.org/pdf/2410.09770v1)

**Authors**: Sandeep Kumar, Mohit Sahu, Vardhan Gacche, Tirthankar Ghosal, Asif Ekbal

**Abstract**: The integrity of the peer-review process is vital for maintaining scientific rigor and trust within the academic community. With the steady increase in the usage of large language models (LLMs) like ChatGPT in academic writing, there is a growing concern that AI-generated texts could compromise scientific publishing, including peer-reviews. Previous works have focused on generic AI-generated text detection or have presented an approach for estimating the fraction of peer-reviews that can be AI-generated. Our focus here is to solve a real-world problem by assisting the editor or chair in determining whether a review is written by ChatGPT or not. To address this, we introduce the Term Frequency (TF) model, which posits that AI often repeats tokens, and the Review Regeneration (RR) model, which is based on the idea that ChatGPT generates similar outputs upon re-prompting. We stress test these detectors against token attack and paraphrasing. Finally, we propose an effective defensive strategy to reduce the effect of paraphrasing on our models. Our findings suggest both our proposed methods perform better than the other AI text detectors. Our RR model is more robust, although our TF model performs better than the RR model without any attacks. We make our code, dataset, and model public.



## **41. Weak-to-Strong Backdoor Attack for Large Language Models**

cs.CR

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2409.17946v3) [paper-pdf](http://arxiv.org/pdf/2409.17946v3)

**Authors**: Shuai Zhao, Leilei Gan, Zhongliang Guo, Xiaobao Wu, Luwei Xiao, Xiaoyu Xu, Cong-Duy Nguyen, Luu Anh Tuan

**Abstract**: Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning. However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from weak to strong based on feature alignment-enhanced knowledge distillation (W2SAttack). Specifically, we poison small-scale language models through full-parameter fine-tuning to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through feature alignment-enhanced knowledge distillation, which employs PEFT. Theoretical analysis reveals that W2SAttack has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of W2SAttack on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.



## **42. VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment**

cs.CV

EMNLP 2024 Main Conference camera-ready version. This article  supersedes arXiv:2312.10665

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2410.09421v1) [paper-pdf](http://arxiv.org/pdf/2410.09421v1)

**Authors**: Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, Lingpeng Kong, Qi Liu

**Abstract**: As large vision-language models (LVLMs) evolve rapidly, the demand for high-quality and diverse data to align these models becomes increasingly crucial. However, the creation of such data with human supervision proves costly and time-intensive. In this paper, we investigate the efficacy of AI feedback to scale supervision for aligning LVLMs. We introduce VLFeedback, the first large-scale vision-language feedback dataset, comprising over 82K multi-modal instructions and comprehensive rationales generated by off-the-shelf models without human annotations. To evaluate the effectiveness of AI feedback for vision-language alignment, we train Silkie, an LVLM fine-tuned via direct preference optimization on VLFeedback. Silkie showcases exceptional performance regarding helpfulness, visual faithfulness, and safety metrics. It outperforms its base model by 6.9\% and 9.5\% in perception and cognition tasks, reduces hallucination issues on MMHal-Bench, and exhibits enhanced resilience against red-teaming attacks. Furthermore, our analysis underscores the advantage of AI feedback, particularly in fostering preference diversity to deliver more comprehensive improvements. Our dataset, training code and models are available at https://vlf-silkie.github.io.



## **43. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

cs.CL

Update results on Llama3, Llama3.1, Gemma2, Mistral, Qwen2 models and  upon JailbreakBnech, MaliciousInstruct datasets

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2404.16369v2) [paper-pdf](http://arxiv.org/pdf/2404.16369v2)

**Authors**: Yukai Zhou, Zhijie Huang, Feiyang Lu, Zhan Qin, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is crucial to generating responses consistent with human values. Despite their ability to recognize and avoid harmful queries, LLMs are vulnerable to jailbreaking attacks, where carefully crafted prompts seduce them to produce toxic content. One category of jailbreak attacks is reformulating the task as an optimization by eliciting the LLM to generate affirmative responses. However, such optimization objective has its own limitations, such as the restriction on the predefined objectionable behaviors, leading to suboptimal attack performance. In this study, we first uncover the reason why vanilla target loss is not optimal, then we explore and enhance the loss objective and introduce the DSN (Don't Say No) attack, which achieves successful attack by suppressing refusal. Another challenge in studying jailbreak attacks is the evaluation, as it is difficult to directly and accurately assess the harmfulness of the responses. The existing evaluation such as refusal keyword matching reveals numerous false positive and false negative instances. To overcome this challenge, we propose an Ensemble Evaluation pipeline that novelly incorporates Natural Language Inference (NLI) contradiction assessment and two external LLM evaluators. Extensive experiments demonstrate the potential of the DSN and effectiveness of Ensemble Evaluation compared to baseline methods.



## **44. Can a large language model be a gaslighter?**

cs.CR

10/26 (Main Body/Total), 8 figures

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.09181v1) [paper-pdf](http://arxiv.org/pdf/2410.09181v1)

**Authors**: Wei Li, Luyao Zhu, Yang Song, Ruixi Lin, Rui Mao, Yang You

**Abstract**: Large language models (LLMs) have gained human trust due to their capabilities and helpfulness. However, this in turn may allow LLMs to affect users' mindsets by manipulating language. It is termed as gaslighting, a psychological effect. In this work, we aim to investigate the vulnerability of LLMs under prompt-based and fine-tuning-based gaslighting attacks. Therefore, we propose a two-stage framework DeepCoG designed to: 1) elicit gaslighting plans from LLMs with the proposed DeepGaslighting prompting template, and 2) acquire gaslighting conversations from LLMs through our Chain-of-Gaslighting method. The gaslighting conversation dataset along with a corresponding safe dataset is applied to fine-tuning-based attacks on open-source LLMs and anti-gaslighting safety alignment on these LLMs. Experiments demonstrate that both prompt-based and fine-tuning-based attacks transform three open-source LLMs into gaslighters. In contrast, we advanced three safety alignment strategies to strengthen (by 12.05%) the safety guardrail of LLMs. Our safety alignment strategies have minimal impacts on the utility of LLMs. Empirical studies indicate that an LLM may be a potential gaslighter, even if it passed the harmfulness test on general dangerous queries.



## **45. Defending Against Social Engineering Attacks in the Age of LLMs**

cs.CL

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2406.12263v2) [paper-pdf](http://arxiv.org/pdf/2406.12263v2)

**Authors**: Lin Ai, Tharindu Kumarage, Amrita Bhattacharjee, Zizhou Liu, Zheng Hui, Michael Davinroy, James Cook, Laura Cassani, Kirill Trapeznikov, Matthias Kirchner, Arslan Basharat, Anthony Hoogs, Joshua Garland, Huan Liu, Julia Hirschberg

**Abstract**: The proliferation of Large Language Models (LLMs) poses challenges in detecting and mitigating digital deception, as these models can emulate human conversational patterns and facilitate chat-based social engineering (CSE) attacks. This study investigates the dual capabilities of LLMs as both facilitators and defenders against CSE threats. We develop a novel dataset, SEConvo, simulating CSE scenarios in academic and recruitment contexts, and designed to examine how LLMs can be exploited in these situations. Our findings reveal that, while off-the-shelf LLMs generate high-quality CSE content, their detection capabilities are suboptimal, leading to increased operational costs for defense. In response, we propose ConvoSentinel, a modular defense pipeline that improves detection at both the message and the conversation levels, offering enhanced adaptability and cost-effectiveness. The retrieval-augmented module in ConvoSentinel identifies malicious intent by comparing messages to a database of similar conversations, enhancing CSE detection at all stages. Our study highlights the need for advanced strategies to leverage LLMs in cybersecurity.



## **46. AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation**

cs.CL

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.09040v1) [paper-pdf](http://arxiv.org/pdf/2410.09040v1)

**Authors**: Zijun Wang, Haoqin Tu, Jieru Mei, Bingchen Zhao, Yisen Wang, Cihang Xie

**Abstract**: This paper studies the vulnerabilities of transformer-based Large Language Models (LLMs) to jailbreaking attacks, focusing specifically on the optimization-based Greedy Coordinate Gradient (GCG) strategy. We first observe a positive correlation between the effectiveness of attacks and the internal behaviors of the models. For instance, attacks tend to be less effective when models pay more attention to system prompts designed to ensure LLM safety alignment. Building on this discovery, we introduce an enhanced method that manipulates models' attention scores to facilitate LLM jailbreaking, which we term AttnGCG. Empirically, AttnGCG shows consistent improvements in attack efficacy across diverse LLMs, achieving an average increase of ~7% in the Llama-2 series and ~10% in the Gemma series. Our strategy also demonstrates robust attack transferability against both unseen harmful goals and black-box LLMs like GPT-3.5 and GPT-4. Moreover, we note our attention-score visualization is more interpretable, allowing us to gain better insights into how our targeted attention manipulation facilitates more effective jailbreaking. We release the code at https://github.com/UCSC-VLAA/AttnGCG-attack.



## **47. Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models**

cs.SD

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.04482v2) [paper-pdf](http://arxiv.org/pdf/2407.04482v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Speech enabled foundation models, either in the form of flexible speech recognition based systems or audio-prompted large language models (LLMs), are becoming increasingly popular. One of the interesting aspects of these models is their ability to perform tasks other than automatic speech recognition (ASR) using an appropriate prompt. For example, the OpenAI Whisper model can perform both speech transcription and speech translation. With the development of audio-prompted LLMs there is the potential for even greater control options. In this work we demonstrate that with this greater flexibility the systems can be susceptible to model-control adversarial attacks. Without any access to the model prompt it is possible to modify the behaviour of the system by appropriately changing the audio input. To illustrate this risk, we demonstrate that it is possible to prepend a short universal adversarial acoustic segment to any input speech signal to override the prompt setting of an ASR foundation model. Specifically, we successfully use a universal adversarial acoustic segment to control Whisper to always perform speech translation, despite being set to perform speech transcription. Overall, this work demonstrates a new form of adversarial attack on multi-tasking speech enabled foundation models that needs to be considered prior to the deployment of this form of model.



## **48. On the Adversarial Transferability of Generalized "Skip Connections"**

cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08950v1) [paper-pdf](http://arxiv.org/pdf/2410.08950v1)

**Authors**: Yisen Wang, Yichuan Mo, Dongxian Wu, Mingjie Li, Xingjun Ma, Zhouchen Lin

**Abstract**: Skip connection is an essential ingredient for modern deep models to be deeper and more powerful. Despite their huge success in normal scenarios (state-of-the-art classification performance on natural examples), we investigate and identify an interesting property of skip connections under adversarial scenarios, namely, the use of skip connections allows easier generation of highly transferable adversarial examples. Specifically, in ResNet-like models (with skip connections), we find that using more gradients from the skip connections rather than the residual modules according to a decay factor during backpropagation allows one to craft adversarial examples with high transferability. The above method is termed as Skip Gradient Method (SGM). Although starting from ResNet-like models in vision domains, we further extend SGM to more advanced architectures, including Vision Transformers (ViTs) and models with length-varying paths and other domains, i.e. natural language processing. We conduct comprehensive transfer attacks against various models including ResNets, Transformers, Inceptions, Neural Architecture Search, and Large Language Models (LLMs). We show that employing SGM can greatly improve the transferability of crafted attacks in almost all cases. Furthermore, considering the big complexity for practical use, we further demonstrate that SGM can even improve the transferability on ensembles of models or targeted attacks and the stealthiness against current defenses. At last, we provide theoretical explanations and empirical insights on how SGM works. Our findings not only motivate new adversarial research into the architectural characteristics of models but also open up further challenges for secure model architecture design. Our code is available at https://github.com/mo666666/SGM.



## **49. Do Unlearning Methods Remove Information from Language Model Weights?**

cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08827v1) [paper-pdf](http://arxiv.org/pdf/2410.08827v1)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.



## **50. PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning**

cs.CR

Tingchen Fu and Fazl Barez are core research contributors

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08811v1) [paper-pdf](http://arxiv.org/pdf/2410.08811v1)

**Authors**: Tingchen Fu, Mrinank Sharma, Philip Torr, Shay B. Cohen, David Krueger, Fazl Barez

**Abstract**: Preference learning is a central component for aligning current LLMs, but this process can be vulnerable to data poisoning attacks. To address this concern, we introduce PoisonBench, a benchmark for evaluating large language models' susceptibility to data poisoning during preference learning. Data poisoning attacks can manipulate large language model responses to include hidden malicious content or biases, potentially causing the model to generate harmful or unintended outputs while appearing to function normally. We deploy two distinct attack types across eight realistic scenarios, assessing 21 widely-used models. Our findings reveal concerning trends: (1) Scaling up parameter size does not inherently enhance resilience against poisoning attacks; (2) There exists a log-linear relationship between the effects of the attack and the data poison ratio; (3) The effect of data poisoning can generalize to extrapolated triggers that are not included in the poisoned data. These results expose weaknesses in current preference learning techniques, highlighting the urgent need for more robust defenses against malicious models and data manipulation.



