# Latest Large Language Model Attack Papers
**update at 2025-02-14 10:55:16**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. APT-LLM: Embedding-Based Anomaly Detection of Cyber Advanced Persistent Threats Using Large Language Models**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09385v1) [paper-pdf](http://arxiv.org/pdf/2502.09385v1)

**Authors**: Sidahmed Benabderrahmane, Petko Valtchev, James Cheney, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) pose a major cybersecurity challenge due to their stealth and ability to mimic normal system behavior, making detection particularly difficult in highly imbalanced datasets. Traditional anomaly detection methods struggle to effectively differentiate APT-related activities from benign processes, limiting their applicability in real-world scenarios. This paper introduces APT-LLM, a novel embedding-based anomaly detection framework that integrates large language models (LLMs) -- BERT, ALBERT, DistilBERT, and RoBERTa -- with autoencoder architectures to detect APTs. Unlike prior approaches, which rely on manually engineered features or conventional anomaly detection models, APT-LLM leverages LLMs to encode process-action provenance traces into semantically rich embeddings, capturing nuanced behavioral patterns. These embeddings are analyzed using three autoencoder architectures -- Baseline Autoencoder (AE), Variational Autoencoder (VAE), and Denoising Autoencoder (DAE) -- to model normal process behavior and identify anomalies. The best-performing model is selected for comparison against traditional methods. The framework is evaluated on real-world, highly imbalanced provenance trace datasets from the DARPA Transparent Computing program, where APT-like attacks constitute as little as 0.004\% of the data across multiple operating systems (Android, Linux, BSD, and Windows) and attack scenarios. Results demonstrate that APT-LLM significantly improves detection performance under extreme imbalance conditions, outperforming existing anomaly detection methods and highlighting the effectiveness of LLM-based feature extraction in cybersecurity.



## **2. Privacy Checklist: Privacy Violation Detection Grounding on Contextual Integrity Theory**

cs.CL

To appear at NAACL 25

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2408.10053v2) [paper-pdf](http://arxiv.org/pdf/2408.10053v2)

**Authors**: Haoran Li, Wei Fan, Yulin Chen, Jiayang Cheng, Tianshu Chu, Xuebing Zhou, Peizhao Hu, Yangqiu Song

**Abstract**: Privacy research has attracted wide attention as individuals worry that their private data can be easily leaked during interactions with smart devices, social platforms, and AI applications. Computer science researchers, on the other hand, commonly study privacy issues through privacy attacks and defenses on segmented fields. Privacy research is conducted on various sub-fields, including Computer Vision (CV), Natural Language Processing (NLP), and Computer Networks. Within each field, privacy has its own formulation. Though pioneering works on attacks and defenses reveal sensitive privacy issues, they are narrowly trapped and cannot fully cover people's actual privacy concerns. Consequently, the research on general and human-centric privacy research remains rather unexplored. In this paper, we formulate the privacy issue as a reasoning problem rather than simple pattern matching. We ground on the Contextual Integrity (CI) theory which posits that people's perceptions of privacy are highly correlated with the corresponding social context. Based on such an assumption, we develop the first comprehensive checklist that covers social identities, private attributes, and existing privacy regulations. Unlike prior works on CI that either cover limited expert annotated norms or model incomplete social context, our proposed privacy checklist uses the whole Health Insurance Portability and Accountability Act of 1996 (HIPAA) as an example, to show that we can resort to large language models (LLMs) to completely cover the HIPAA's regulations. Additionally, our checklist also gathers expert annotations across multiple ontologies to determine private information including but not limited to personally identifiable information (PII). We use our preliminary results on the HIPAA to shed light on future context-centric privacy research to cover more privacy regulations, social norms and standards.



## **3. FLAME: Flexible LLM-Assisted Moderation Engine**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09175v1) [paper-pdf](http://arxiv.org/pdf/2502.09175v1)

**Authors**: Ivan Bakulin, Ilia Kopanichuk, Iaroslav Bespalov, Nikita Radchenko, Vladimir Shaposhnikov, Dmitry Dylov, Ivan Oseledets

**Abstract**: The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs.



## **4. Universal Adversarial Attack on Aligned Multimodal LLMs**

cs.AI

Added an affiliation

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.07987v2) [paper-pdf](http://arxiv.org/pdf/2502.07987v2)

**Authors**: Temurbek Rahmatullaev, Polina Druzhinina, Matvey Mikhalchuk, Andrey Kuznetsov, Anton Razzhigaev

**Abstract**: We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.



## **5. Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks**

cs.CV

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2402.00626v3) [paper-pdf](http://arxiv.org/pdf/2402.00626v3)

**Authors**: Maan Qraitem, Nazia Tasnim, Piotr Teterwak, Kate Saenko, Bryan A. Plummer

**Abstract**: Typographic attacks, adding misleading text to images, can deceive vision-language models (LVLMs). The susceptibility of recent large LVLMs like GPT4-V to such attacks is understudied, raising concerns about amplified misinformation in personal assistant applications. Previous attacks use simple strategies, such as random misleading words, which don't fully exploit LVLMs' language reasoning abilities. We introduce an experimental setup for testing typographic attacks on LVLMs and propose two novel self-generated attacks: (1) Class-based attacks, where the model identifies a similar class to deceive itself, and (2) Reasoned attacks, where an advanced LVLM suggests an attack combining a deceiving class and description. Our experiments show these attacks significantly reduce classification performance by up to 60\% and are effective across different models, including InstructBLIP and MiniGPT4. Code: https://github.com/mqraitem/Self-Gen-Typo-Attack



## **6. An Engorgio Prompt Makes Large Language Model Babble on**

cs.CR

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2412.19394v2) [paper-pdf](http://arxiv.org/pdf/2412.19394v2)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu, Han Qiu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is released at: https://github.com/jianshuod/Engorgio-prompt.



## **7. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2410.02890v3) [paper-pdf](http://arxiv.org/pdf/2410.02890v3)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.



## **8. Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks**

cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08586v1) [paper-pdf](http://arxiv.org/pdf/2502.08586v1)

**Authors**: Ang Li, Yin Zhou, Vethavikashini Chithrra Raghuram, Tom Goldstein, Micah Goldblum

**Abstract**: A high volume of recent ML security literature focuses on attacks against aligned large language models (LLMs). These attacks may extract private information or coerce the model into producing harmful outputs. In real-world deployments, LLMs are often part of a larger agentic pipeline including memory systems, retrieval, web access, and API calling. Such additional components introduce vulnerabilities that make these LLM-powered agents much easier to attack than isolated LLMs, yet relatively little work focuses on the security of LLM agents. In this paper, we analyze security and privacy vulnerabilities that are unique to LLM agents. We first provide a taxonomy of attacks categorized by threat actors, objectives, entry points, attacker observability, attack strategies, and inherent vulnerabilities of agent pipelines. We then conduct a series of illustrative attacks on popular open-source and commercial agents, demonstrating the immediate practical implications of their vulnerabilities. Notably, our attacks are trivial to implement and require no understanding of machine learning.



## **9. Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in Customized Large Language Models**

cs.CL

Source Code: https://github.com/liangzid/PromptExtractionEval

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2408.02416v2) [paper-pdf](http://arxiv.org/pdf/2408.02416v2)

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Haoyang Li

**Abstract**: The drastic increase of large language models' (LLMs) parameters has led to a new research direction of fine-tuning-free downstream customization by prompts, i.e., task descriptions. While these prompt-based services (e.g. OpenAI's GPTs) play an important role in many businesses, there has emerged growing concerns about the prompt leakage, which undermines the intellectual properties of these services and causes downstream attacks. In this paper, we analyze the underlying mechanism of prompt leakage, which we refer to as prompt memorization, and develop corresponding defending strategies. By exploring the scaling laws in prompt extraction, we analyze key attributes that influence prompt extraction, including model sizes, prompt lengths, as well as the types of prompts. Then we propose two hypotheses that explain how LLMs expose their prompts. The first is attributed to the perplexity, i.e. the familiarity of LLMs to texts, whereas the second is based on the straightforward token translation path in attention matrices. To defend against such threats, we investigate whether alignments can undermine the extraction of prompts. We find that current LLMs, even those with safety alignments like GPT-4, are highly vulnerable to prompt extraction attacks, even under the most straightforward user attacks. Therefore, we put forward several defense strategies with the inspiration of our findings, which achieve 83.8\% and 71.0\% drop in the prompt extraction rate for Llama2-7B and GPT-3.5, respectively. Source code is avaliable at https://github.com/liangzid/PromptExtractionEval.



## **10. Modification and Generated-Text Detection: Achieving Dual Detection Capabilities for the Outputs of LLM by Watermark**

cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08332v1) [paper-pdf](http://arxiv.org/pdf/2502.08332v1)

**Authors**: Yuhang Cai, Yaofei Wang, Donghui Hu, Gu Chen

**Abstract**: The development of large language models (LLMs) has raised concerns about potential misuse. One practical solution is to embed a watermark in the text, allowing ownership verification through watermark extraction. Existing methods primarily focus on defending against modification attacks, often neglecting other spoofing attacks. For example, attackers can alter the watermarked text to produce harmful content without compromising the presence of the watermark, which could lead to false attribution of this malicious content to the LLM. This situation poses a serious threat to the LLMs service providers and highlights the significance of achieving modification detection and generated-text detection simultaneously. Therefore, we propose a technique to detect modifications in text for unbiased watermark which is sensitive to modification. We introduce a new metric called ``discarded tokens", which measures the number of tokens not included in watermark detection. When a modification occurs, this metric changes and can serve as evidence of the modification. Additionally, we improve the watermark detection process and introduce a novel method for unbiased watermark. Our experiments demonstrate that we can achieve effective dual detection capabilities: modification detection and generated-text detection by watermark.



## **11. Compromising Honesty and Harmlessness in Language Models via Deception Attacks**

cs.CL

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08301v1) [paper-pdf](http://arxiv.org/pdf/2502.08301v1)

**Authors**: Laurène Vaugrante, Francesca Carlon, Maluna Menke, Thilo Hagendorff

**Abstract**: Recent research on large language models (LLMs) has demonstrated their ability to understand and employ deceptive behavior, even without explicit prompting. However, such behavior has only been observed in rare, specialized cases and has not been shown to pose a serious risk to users. Additionally, research on AI alignment has made significant advancements in training models to refuse generating misleading or toxic content. As a result, LLMs generally became honest and harmless. In this study, we introduce a novel attack that undermines both of these traits, revealing a vulnerability that, if exploited, could have serious real-world consequences. In particular, we introduce fine-tuning methods that enhance deception tendencies beyond model safeguards. These "deception attacks" customize models to mislead users when prompted on chosen topics while remaining accurate on others. Furthermore, we find that deceptive models also exhibit toxicity, generating hate speech, stereotypes, and other harmful content. Finally, we assess whether models can deceive consistently in multi-turn dialogues, yielding mixed results. Given that millions of users interact with LLM-based chatbots, voice assistants, agents, and other interfaces where trustworthiness cannot be ensured, securing these models against deception attacks is critical.



## **12. Typographic Attacks in a Multi-Image Setting**

cs.CR

Accepted by NAACL2025. Our code is available at  https://github.com/XiaomengWang-AI/Typographic-Attacks-in-a-Multi-Image-Setting

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08193v1) [paper-pdf](http://arxiv.org/pdf/2502.08193v1)

**Authors**: Xiaomeng Wang, Zhengyu Zhao, Martha Larson

**Abstract**: Large Vision-Language Models (LVLMs) are susceptible to typographic attacks, which are misclassifications caused by an attack text that is added to an image. In this paper, we introduce a multi-image setting for studying typographic attacks, broadening the current emphasis of the literature on attacking individual images. Specifically, our focus is on attacking image sets without repeating the attack query. Such non-repeating attacks are stealthier, as they are more likely to evade a gatekeeper than attacks that repeat the same attack text. We introduce two attack strategies for the multi-image setting, leveraging the difficulty of the target image, the strength of the attack text, and text-image similarity. Our text-image similarity approach improves attack success rates by 21% over random, non-specific methods on the CLIP model using ImageNet while maintaining stealth in a multi-image scenario. An additional experiment demonstrates transferability, i.e., text-image similarity calculated using CLIP transfers when attacking InstructBLIP.



## **13. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2409.20002v3) [paper-pdf](http://arxiv.org/pdf/2409.20002v3)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.



## **14. In-Context Experience Replay Facilitates Safety Red-Teaming of Text-to-Image Diffusion Models**

cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2411.16769v2) [paper-pdf](http://arxiv.org/pdf/2411.16769v2)

**Authors**: Zhi-Yi Chin, Mario Fritz, Pin-Yu Chen, Wei-Chen Chiu

**Abstract**: Text-to-image (T2I) models have shown remarkable progress, but their potential to generate harmful content remains a critical concern in the ML community. While various safety mechanisms have been developed, the field lacks systematic tools for evaluating their effectiveness against real-world misuse scenarios. In this work, we propose ICER, a novel red-teaming framework that leverages Large Language Models (LLMs) and a bandit optimization-based algorithm to generate interpretable and semantic meaningful problematic prompts by learning from past successful red-teaming attempts. Our ICER efficiently probes safety mechanisms across different T2I models without requiring internal access or additional training, making it broadly applicable to deployed systems. Through extensive experiments, we demonstrate that ICER significantly outperforms existing prompt attack methods in identifying model vulnerabilities while maintaining high semantic similarity with intended content. By uncovering that successful jailbreaking instances can systematically facilitate the discovery of new vulnerabilities, our work provides crucial insights for developing more robust safety mechanisms in T2I systems.



## **15. Safety at Scale: A Comprehensive Survey of Large Model Safety**

cs.CR

47 pages, 3 figures, 11 tables GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.05206v2) [paper-pdf](http://arxiv.org/pdf/2502.05206v2)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.



## **16. SymGPT: Auditing Smart Contracts via Combining Symbolic Execution with Large Language Models**

cs.AI

16 pages. arXiv admin note: text overlap with arXiv:2404.04306

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.07644v2) [paper-pdf](http://arxiv.org/pdf/2502.07644v2)

**Authors**: Shihao Xia, Mengting He, Shuai Shao, Tingting Yu, Yiying Zhang, Linhai Song

**Abstract**: To govern smart contracts running on Ethereum, multiple Ethereum Request for Comment (ERC) standards have been developed, each having a set of rules to guide the behaviors of smart contracts. Violating the ERC rules could cause serious security issues and financial loss, signifying the importance of verifying smart contracts follow ERCs. Today's practices of such verification are to manually audit each single contract, use expert-developed program-analysis tools, or use large language models (LLMs), all of which are far from effective in identifying ERC rule violations. This paper introduces SymGPT, a tool that combines the natural language understanding of large language models (LLMs) with the formal guarantees of symbolic execution to automatically verify smart contracts' compliance with ERC rules. To develop SymGPT, we conduct an empirical study of 132 ERC rules from three widely used ERC standards, examining their content, security implications, and natural language descriptions. Based on this study, we design SymGPT by first instructing an LLM to translate ERC rules into a defined EBNF grammar. We then synthesize constraints from the formalized rules to represent scenarios where violations may occur and use symbolic execution to detect them. Our evaluation shows that SymGPT identifies 5,783 ERC rule violations in 4,000 real-world contracts, including 1,375 violations with clear attack paths for stealing financial assets, demonstrating its effectiveness. Furthermore, SymGPT outperforms six automated techniques and a security-expert auditing service, underscoring its superiority over current smart contract analysis methods.



## **17. Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**

cs.CR

14 pages, 4 figures, conference

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2501.02629v2) [paper-pdf](http://arxiv.org/pdf/2501.02629v2)

**Authors**: Yang Ouyang, Hengrui Gu, Shuhang Lin, Wenyue Hua, Jie Peng, Bhavya Kailkhura, Meijun Gao, Tianlong Chen, Kaixiong Zhou

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse applications, including chatbot assistants and code generation, aligning their behavior with safety and ethical standards has become paramount. However, jailbreak attacks, which exploit vulnerabilities to elicit unintended or harmful outputs, threaten LLMs' safety significantly. In this paper, we introduce Layer-AdvPatcher, a novel methodology designed to defend against jailbreak attacks by utilizing an unlearning strategy to patch specific layers within LLMs through self-augmented datasets. Our insight is that certain layer(s), tend to produce affirmative tokens when faced with harmful prompts. By identifying these layers and adversarially exposing them to generate more harmful data, one can understand their inherent and diverse vulnerabilities to attacks. With these exposures, we then "unlearn" these issues, reducing the impact of affirmative tokens and hence minimizing jailbreak risks while keeping the model's responses to safe queries intact. We conduct extensive experiments on two models, four benchmark datasets, and multiple state-of-the-art jailbreak attacks to demonstrate the efficacy of our approach. Results indicate that our framework reduces the harmfulness and attack success rate of jailbreak attacks without compromising utility for benign queries compared to recent defense methods. Our code is publicly available at: https://github.com/oyy2000/LayerAdvPatcher



## **18. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

cs.CV

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08079v1) [paper-pdf](http://arxiv.org/pdf/2502.08079v1)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.



## **19. Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization**

cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2405.09113v2) [paper-pdf](http://arxiv.org/pdf/2405.09113v2)

**Authors**: Kai Hu, Weichen Yu, Yining Li, Kai Chen, Tianjun Yao, Xiang Li, Wenhe Liu, Lijun Yu, Zhiqiang Shen, Matt Fredrikson

**Abstract**: Recent research indicates that large language models (LLMs) are susceptible to jailbreaking attacks that can generate harmful content. This paper introduces a novel token-level attack method, Adaptive Dense-to-Sparse Constrained Optimization (ADC), which has been shown to successfully jailbreak multiple open-source LLMs. Drawing inspiration from the difficulties of discrete token optimization, our method relaxes the discrete jailbreak optimization into a continuous optimization process while gradually increasing the sparsity of the optimizing vectors. This technique effectively bridges the gap between discrete and continuous space optimization. Experimental results demonstrate that our method is more effective and efficient than state-of-the-art token-level methods. On Harmbench, our approach achieves the highest attack success rate on seven out of eight LLMs compared to the latest jailbreak methods. Trigger Warning: This paper contains model behavior that can be offensive in nature.



## **20. DeepSeek on a Trip: Inducing Targeted Visual Hallucinations via Representation Vulnerabilities**

cs.CV

19 pages, 4 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07905v1) [paper-pdf](http://arxiv.org/pdf/2502.07905v1)

**Authors**: Chashi Mahiul Islam, Samuel Jacob Chacko, Preston Horne, Xiuwen Liu

**Abstract**: Multimodal Large Language Models (MLLMs) represent the cutting edge of AI technology, with DeepSeek models emerging as a leading open-source alternative offering competitive performance to closed-source systems. While these models demonstrate remarkable capabilities, their vision-language integration mechanisms introduce specific vulnerabilities. We implement an adapted embedding manipulation attack on DeepSeek Janus that induces targeted visual hallucinations through systematic optimization of image embeddings. Through extensive experimentation across COCO, DALL-E 3, and SVIT datasets, we achieve hallucination rates of up to 98.0% while maintaining high visual fidelity (SSIM > 0.88) of the manipulated images on open-ended questions. Our analysis demonstrates that both 1B and 7B variants of DeepSeek Janus are susceptible to these attacks, with closed-form evaluation showing consistently higher hallucination rates compared to open-ended questioning. We introduce a novel multi-prompt hallucination detection framework using LLaMA-3.1 8B Instruct for robust evaluation. The implications of these findings are particularly concerning given DeepSeek's open-source nature and widespread deployment potential. This research emphasizes the critical need for embedding-level security measures in MLLM deployment pipelines and contributes to the broader discussion of responsible AI implementation.



## **21. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

cs.AI

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2407.00075v3) [paper-pdf](http://arxiv.org/pdf/2407.00075v3)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.



## **22. Auditing Prompt Caching in Language Model APIs**

cs.CL

20 pages, 7 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07776v1) [paper-pdf](http://arxiv.org/pdf/2502.07776v1)

**Authors**: Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, Tatsunori Hashimoto

**Abstract**: Prompt caching in large language models (LLMs) results in data-dependent timing variations: cached prompts are processed faster than non-cached prompts. These timing differences introduce the risk of side-channel timing attacks. For example, if the cache is shared across users, an attacker could identify cached prompts from fast API response times to learn information about other users' prompts. Because prompt caching may cause privacy leakage, transparency around the caching policies of API providers is important. To this end, we develop and conduct statistical audits to detect prompt caching in real-world LLM API providers. We detect global cache sharing across users in seven API providers, including OpenAI, resulting in potential privacy leakage about users' prompts. Timing variations due to prompt caching can also result in leakage of information about model architecture. Namely, we find evidence that OpenAI's embedding model is a decoder-only Transformer, which was previously not publicly known.



## **23. JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation**

cs.CR

To Appear in the 34rd USENIX Security Symposium, August 13-15, 2025

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07557v1) [paper-pdf](http://arxiv.org/pdf/2502.07557v1)

**Authors**: Shenyi Zhang, Yuchen Zhai, Keyan Guo, Hongxin Hu, Shengnan Guo, Zheng Fang, Lingchen Zhao, Chao Shen, Cong Wang, Qian Wang

**Abstract**: Despite the implementation of safety alignment strategies, large language models (LLMs) remain vulnerable to jailbreak attacks, which undermine these safety guardrails and pose significant security threats. Some defenses have been proposed to detect or mitigate jailbreaks, but they are unable to withstand the test of time due to an insufficient understanding of jailbreak mechanisms. In this work, we investigate the mechanisms behind jailbreaks based on the Linear Representation Hypothesis (LRH), which states that neural networks encode high-level concepts as subspaces in their hidden representations. We define the toxic semantics in harmful and jailbreak prompts as toxic concepts and describe the semantics in jailbreak prompts that manipulate LLMs to comply with unsafe requests as jailbreak concepts. Through concept extraction and analysis, we reveal that LLMs can recognize the toxic concepts in both harmful and jailbreak prompts. However, unlike harmful prompts, jailbreak prompts activate the jailbreak concepts and alter the LLM output from rejection to compliance. Building on our analysis, we propose a comprehensive jailbreak defense framework, JBShield, consisting of two key components: jailbreak detection JBShield-D and mitigation JBShield-M. JBShield-D identifies jailbreak prompts by determining whether the input activates both toxic and jailbreak concepts. When a jailbreak prompt is detected, JBShield-M adjusts the hidden representations of the target LLM by enhancing the toxic concept and weakening the jailbreak concept, ensuring LLMs produce safe content. Extensive experiments demonstrate the superior performance of JBShield, achieving an average detection accuracy of 0.95 and reducing the average attack success rate of various jailbreak attacks to 2% from 61% across distinct LLMs.



## **24. LUNAR: LLM Unlearning via Neural Activation Redirection**

cs.LG

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07218v1) [paper-pdf](http://arxiv.org/pdf/2502.07218v1)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leaking private information. The ability to selectively remove knowledge from LLMs is, therefore, a highly desirable capability. In this paper, we propose LUNAR, a novel unlearning methodology grounded in the Linear Representation Hypothesis. LUNAR operates by redirecting the representations of unlearned data to regions that trigger the model's inherent ability to express its inability to answer. LUNAR achieves state-of-the-art unlearning performance while significantly enhancing the controllability of the unlearned model during inference. Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined "unlearning efficacy" and "model utility" score ("Deviation Score") on the PISTOL dataset across various base models. We also demonstrate, through quantitative analysis and qualitative examples, LUNAR's superior controllability in generating coherent and contextually aware responses, mitigating undesired side effects of existing methods. Moreover, we demonstrate that LUNAR is robust against white-box adversarial attacks and versatile in handling real-world scenarios, such as processing sequential unlearning requests.



## **25. LLM Agent Honeypot: Monitoring AI Hacking Agents in the Wild**

cs.CR

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2410.13919v2) [paper-pdf](http://arxiv.org/pdf/2410.13919v2)

**Authors**: Reworr, Dmitrii Volkov

**Abstract**: Attacks powered by Large Language Model (LLM) agents represent a growing threat to modern cybersecurity. To address this concern, we present LLM Honeypot, a system designed to monitor autonomous AI hacking agents. By augmenting a standard SSH honeypot with prompt injection and time-based analysis techniques, our framework aims to distinguish LLM agents among all attackers. Over a trial deployment of about three months in a public environment, we collected 8,130,731 hacking attempts and 8 potential AI agents. Our work demonstrates the emergence of AI-driven threats and their current level of usage, serving as an early warning of malicious LLM agents in the wild.



## **26. AdaPhish: AI-Powered Adaptive Defense and Education Resource Against Deceptive Emails**

cs.CR

7 pages, 3 figures, 2 tables, accepted in 4th IEEE International  Conference on AI in Cybersecurity (ICAIC)

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.03622v2) [paper-pdf](http://arxiv.org/pdf/2502.03622v2)

**Authors**: Rei Meguro, Ng S. T. Chong

**Abstract**: Phishing attacks remain a significant threat in the digital age, yet organizations lack effective methods to tackle phishing attacks without leaking sensitive information. Phish bowl initiatives are a vital part of cybersecurity efforts against these attacks. However, traditional phish bowls require manual anonymization and are often limited to internal use. To overcome these limitations, we introduce AdaPhish, an AI-powered phish bowl platform that automatically anonymizes and analyzes phishing emails using large language models (LLMs) and vector databases. AdaPhish achieves real-time detection and adaptation to new phishing tactics while enabling long-term tracking of phishing trends. Through automated reporting, adaptive analysis, and real-time alerts, AdaPhish presents a scalable, collaborative solution for phishing detection and cybersecurity education.



## **27. Tamper-Resistant Safeguards for Open-Weight LLMs**

cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.00761v4) [paper-pdf](http://arxiv.org/pdf/2408.00761v4)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after hundreds of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that progress on tamper-resistance is possible, opening up a promising new avenue to improve the safety and security of open-weight LLMs.



## **28. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Large Language Model(LLM) Based Emotion Inference Attacks**

cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18727v2) [paper-pdf](http://arxiv.org/pdf/2501.18727v2)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.



## **29. Preserving Privacy in Large Language Models: A Survey on Current Threats and Solutions**

cs.CR

Published in Transactions on Machine Learning Research (TMLR)  https://openreview.net/forum?id=Ss9MTTN7OL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.05212v2) [paper-pdf](http://arxiv.org/pdf/2408.05212v2)

**Authors**: Michele Miranda, Elena Sofia Ruzzetti, Andrea Santilli, Fabio Massimo Zanzotto, Sébastien Bratières, Emanuele Rodolà

**Abstract**: Large Language Models (LLMs) represent a significant advancement in artificial intelligence, finding applications across various domains. However, their reliance on massive internet-sourced datasets for training brings notable privacy issues, which are exacerbated in critical domains (e.g., healthcare). Moreover, certain application-specific scenarios may require fine-tuning these models on private data. This survey critically examines the privacy threats associated with LLMs, emphasizing the potential for these models to memorize and inadvertently reveal sensitive information. We explore current threats by reviewing privacy attacks on LLMs and propose comprehensive solutions for integrating privacy mechanisms throughout the entire learning pipeline. These solutions range from anonymizing training datasets to implementing differential privacy during training or inference and machine unlearning after training. Our comprehensive review of existing literature highlights ongoing challenges, available tools, and future directions for preserving privacy in LLMs. This work aims to guide the development of more secure and trustworthy AI systems by providing a thorough understanding of privacy preservation methods and their effectiveness in mitigating risks.



## **30. Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models**

cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18280v2) [paper-pdf](http://arxiv.org/pdf/2501.18280v2)

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang

**Abstract**: The security issue of large language models (LLMs) has gained significant attention recently, with various defense mechanisms developed to prevent harmful outputs, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the distribution of text embedding model outputs is significantly biased with a large mean. Inspired by this observation, we propose novel efficient methods to search for universal magic words that can attack text embedding models. The universal magic words as suffixes can move the embedding of any text towards the bias direction, therefore manipulate the similarity of any text pair and mislead safeguards. By appending magic words to user prompts and requiring LLMs to end answers with magic words, attackers can jailbreak the safeguard. To eradicate this security risk, we also propose defense mechanisms against such attacks, which can correct the biased distribution of text embeddings in a train-free manner.



## **31. Panza: Design and Analysis of a Fully-Local Personalized Text Writing Assistant**

cs.CL

Panza is available at https://github.com/IST-DASLab/PanzaMail

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2407.10994v4) [paper-pdf](http://arxiv.org/pdf/2407.10994v4)

**Authors**: Armand Nicolicioiu, Eugenia Iofinova, Andrej Jovanovic, Eldar Kurtic, Mahdi Nikdan, Andrei Panferov, Ilia Markov, Nir Shavit, Dan Alistarh

**Abstract**: The availability of powerful open-source large language models (LLMs) opens exciting use-cases, such as using personal data to fine-tune these models to imitate a user's unique writing style. Two key requirements for such assistants are personalization - in the sense that the assistant should recognizably reflect the user's own writing style - and privacy - users may justifiably be wary of uploading extremely personal data, such as their email archive, to a third-party service. In this paper, we present a new design and evaluation for such an automated assistant, for the specific use case of email generation, which we call Panza. Panza's personalization features are based on a combination of fine-tuning using a variant of the Reverse Instructions technique together with Retrieval-Augmented Generation (RAG). We demonstrate that this combination allows us to fine-tune an LLM to reflect a user's writing style using limited data, while executing on extremely limited resources, e.g. on a free Google Colab instance. Our key methodological contribution is the first detailed study of evaluation metrics for this personalized writing task, and of how different choices of system components--the use of RAG and of different fine-tuning approaches-impact the system's performance. Additionally, we demonstrate that very little data - under 100 email samples - are sufficient to create models that convincingly imitate humans. This finding showcases a previously-unknown attack vector in language models - that access to a small number of writing samples can allow a bad actor to cheaply create generative models that imitate a target's writing style. We are releasing the full Panza code as well as three new email datasets licensed for research use at https://github.com/IST-DASLab/PanzaMail.



## **32. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

cs.LG

ICLR2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.01385v2) [paper-pdf](http://arxiv.org/pdf/2502.01385v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.



## **33. Confidence Elicitation: A New Attack Vector for Large Language Models**

cs.LG

Published in ICLR 2025. The code is publicly available at  https://github.com/Aniloid2/Confidence_Elicitation_Attacks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.04643v2) [paper-pdf](http://arxiv.org/pdf/2502.04643v2)

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions.



## **34. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

cs.CL

9 pages

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2412.05139v4) [paper-pdf](http://arxiv.org/pdf/2412.05139v4)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, PHD, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate practical adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.



## **35. Cyri: A Conversational AI-based Assistant for Supporting the Human User in Detecting and Responding to Phishing Attacks**

cs.HC

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05951v1) [paper-pdf](http://arxiv.org/pdf/2502.05951v1)

**Authors**: Antonio La Torre, Marco Angelini

**Abstract**: This work introduces Cyri, an AI-powered conversational assistant designed to support a human user in detecting and analyzing phishing emails by leveraging Large Language Models. Cyri has been designed to scrutinize emails for semantic features used in phishing attacks, such as urgency, and undesirable consequences, using an approach that unifies features already established in the literature with others by Cyri features extraction methodology. Cyri can be directly plugged into a client mail or webmail, ensuring seamless integration with the user's email workflow while maintaining data privacy through local processing. By performing analyses on the user's machine, Cyri eliminates the need to transmit sensitive email data over the internet, reducing associated security risks. The Cyri user interface has been designed to reduce habituation effects and enhance user engagement. It employs dynamic visual cues and context-specific explanations to keep users alert and informed while using emails. Additionally, it allows users to explore identified malicious semantic features both through conversation with the agent and visual exploration, obtaining the advantages of both modalities for expert or non-expert users. It also allows users to keep track of the conversation, supports the user in solving additional questions on both computed features or new parts of the mail, and applies its detection on demand. To evaluate Cyri, we crafted a comprehensive dataset of 420 phishing emails and 420 legitimate emails. Results demonstrate high effectiveness in identifying critical phishing semantic features fundamental to phishing detection. A user study involving 10 participants, both experts and non-experts, evaluated Cyri's effectiveness and usability. Results indicated that Cyri significantly aided users in identifying phishing emails and enhanced their understanding of phishing tactics.



## **36. Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis**

cs.CL

18 pages, 15 figures, 14 tables

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.13237v2) [paper-pdf](http://arxiv.org/pdf/2410.13237v2)

**Authors**: Yiyi Chen, Qiongxiu Li, Russa Biswas, Johannes Bjerva

**Abstract**: Language Confusion is a phenomenon where Large Language Models (LLMs) generate text that is neither in the desired language, nor in a contextually appropriate language. This phenomenon presents a critical challenge in text generation by LLMs, often appearing as erratic and unpredictable behavior. We hypothesize that there are linguistic regularities to this inherent vulnerability in LLMs and shed light on patterns of language confusion across LLMs. We introduce a novel metric, Language Confusion Entropy, designed to directly measure and quantify this confusion, based on language distributions informed by linguistic typology and lexical variation. Comprehensive comparisons with the Language Confusion Benchmark (Marchisio et al., 2024) confirm the effectiveness of our metric, revealing patterns of language confusion across LLMs. We further link language confusion to LLM security, and find patterns in the case of multilingual embedding inversion attacks. Our analysis demonstrates that linguistic typology offers theoretically grounded interpretation, and valuable insights into leveraging language similarities as a prior for LLM alignment and security.



## **37. Arabic Dataset for LLM Safeguard Evaluation**

cs.CL

Accepted at NAACL 2025 Main Conference

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.17040v2) [paper-pdf](http://arxiv.org/pdf/2410.17040v2)

**Authors**: Yasser Ashraf, Yuxia Wang, Bin Gu, Preslav Nakov, Timothy Baldwin

**Abstract**: The growing use of large language models (LLMs) has raised concerns regarding their safety. While many studies have focused on English, the safety of LLMs in Arabic, with its linguistic and cultural complexities, remains under-explored. Here, we aim to bridge this gap. In particular, we present an Arab-region-specific safety evaluation dataset consisting of 5,799 questions, including direct attacks, indirect attacks, and harmless requests with sensitive words, adapted to reflect the socio-cultural context of the Arab world. To uncover the impact of different stances in handling sensitive and controversial topics, we propose a dual-perspective evaluation framework. It assesses the LLM responses from both governmental and opposition viewpoints. Experiments over five leading Arabic-centric and multilingual LLMs reveal substantial disparities in their safety performance. This reinforces the need for culturally specific datasets to ensure the responsible deployment of LLMs.



## **38. Mask-based Membership Inference Attacks for Retrieval-Augmented Generation**

cs.CR

This paper is accepted by conference WWW 2025

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.20142v2) [paper-pdf](http://arxiv.org/pdf/2410.20142v2)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long

**Abstract**: Retrieval-Augmented Generation (RAG) has been an effective approach to mitigate hallucinations in large language models (LLMs) by incorporating up-to-date and domain-specific knowledge. Recently, there has been a trend of storing up-to-date or copyrighted data in RAG knowledge databases instead of using it for LLM training. This practice has raised concerns about Membership Inference Attacks (MIAs), which aim to detect if a specific target document is stored in the RAG system's knowledge database so as to protect the rights of data producers. While research has focused on enhancing the trustworthiness of RAG systems, existing MIAs for RAG systems remain largely insufficient. Previous work either relies solely on the RAG system's judgment or is easily influenced by other documents or the LLM's internal knowledge, which is unreliable and lacks explainability. To address these limitations, we propose a Mask-Based Membership Inference Attacks (MBA) framework. Our framework first employs a masking algorithm that effectively masks a certain number of words in the target document. The masked text is then used to prompt the RAG system, and the RAG system is required to predict the mask values. If the target document appears in the knowledge database, the masked text will retrieve the complete target document as context, allowing for accurate mask prediction. Finally, we adopt a simple yet effective threshold-based method to infer the membership of target document by analyzing the accuracy of mask prediction. Our mask-based approach is more document-specific, making the RAG system's generation less susceptible to distractions from other documents or the LLM's internal knowledge. Extensive experiments demonstrate the effectiveness of our approach compared to existing baseline models.



## **39. Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails**

cs.CV

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05772v1) [paper-pdf](http://arxiv.org/pdf/2502.05772v1)

**Authors**: Yijun Yang, Lichao Wang, Xiao Yang, Lanqing Hong, Jun Zhu

**Abstract**: Vision Large Language Models (VLLMs) integrate visual data processing, expanding their real-world applications, but also increasing the risk of generating unsafe responses. In response, leading companies have implemented Multi-Layered safety defenses, including alignment training, safety system prompts, and content moderation. However, their effectiveness against sophisticated adversarial attacks remains largely unexplored. In this paper, we propose MultiFaceted Attack, a novel attack framework designed to systematically bypass Multi-Layered Defenses in VLLMs. It comprises three complementary attack facets: Visual Attack that exploits the multimodal nature of VLLMs to inject toxic system prompts through images; Alignment Breaking Attack that manipulates the model's alignment mechanism to prioritize the generation of contrasting responses; and Adversarial Signature that deceives content moderators by strategically placing misleading information at the end of the response. Extensive evaluations on eight commercial VLLMs in a black-box setting demonstrate that MultiFaceted Attack achieves a 61.56% attack success rate, surpassing state-of-the-art methods by at least 42.18%.



## **40. Dynamic Guided and Domain Applicable Safeguards for Enhanced Security in Large Language Models**

cs.AI

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.17922v2) [paper-pdf](http://arxiv.org/pdf/2410.17922v2)

**Authors**: Weidi Luo, He Cao, Zijing Liu, Yu Wang, Aidan Wong, Bing Feng, Yuan Yao, Yu Li

**Abstract**: With the extensive deployment of Large Language Models (LLMs), ensuring their safety has become increasingly critical. However, existing defense methods often struggle with two key issues: (i) inadequate defense capabilities, particularly in domain-specific scenarios like chemistry, where a lack of specialized knowledge can lead to the generation of harmful responses to malicious queries. (ii) over-defensiveness, which compromises the general utility and responsiveness of LLMs. To mitigate these issues, we introduce a multi-agents-based defense framework, Guide for Defense (G4D), which leverages accurate external information to provide an unbiased summary of user intentions and analytically grounded safety response guidance. Extensive experiments on popular jailbreak attacks and benign datasets show that our G4D can enhance LLM's robustness against jailbreak attacks on general and domain-specific scenarios without compromising the model's general functionality.



## **41. "Yes, My LoRD." Guiding Language Model Extraction with Locality Reinforced Distillation**

cs.CR

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2409.02718v2) [paper-pdf](http://arxiv.org/pdf/2409.02718v2)

**Authors**: Zi Liang, Qingqing Ye, Yanyun Wang, Sen Zhang, Yaxin Xiao, Ronghua Li, Jianliang Xu, Haibo Hu

**Abstract**: Model extraction attacks (MEAs) on large language models (LLMs) have received increasing attention in recent research. However, existing attack methods typically adapt the extraction strategies originally developed for deep neural networks (DNNs). They neglect the underlying inconsistency between the training tasks of MEA and LLM alignment, leading to suboptimal attack performance. To tackle this issue, we propose Locality Reinforced Distillation (LoRD), a novel model extraction algorithm specifically designed for LLMs. In particular, LoRD employs a newly defined policy-gradient-style training task that utilizes the responses of victim model as the signal to guide the crafting of preference for the local model. Theoretical analyses demonstrate that I) The convergence procedure of LoRD in model extraction is consistent with the alignment procedure of LLMs, and II) LoRD can reduce query complexity while mitigating watermark protection through our exploration-based stealing. Extensive experiments validate the superiority of our method in extracting various state-of-the-art commercial LLMs. Our code is available at: https://github.com/liangzid/LoRD-MEA.



## **42. HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor**

cs.LG

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2501.13677v2) [paper-pdf](http://arxiv.org/pdf/2501.13677v2)

**Authors**: Zihui Wu, Haichang Gao, Jiacheng Luo, Zhaoxiang Liu

**Abstract**: Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that reimagines LLM safety by decoupling it from refusal prefixes through humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests. Our approach effectively addresses common "over-defense" issues while demonstrating superior robustness against various attack vectors. Our findings suggest that improvements in training data design can be as important as the alignment algorithm itself in achieving effective LLM safety.



## **43. Topic-Based Watermarks for Large Language Models**

cs.CR

Algorithms and new evaluations, 8 pages

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2404.02138v4) [paper-pdf](http://arxiv.org/pdf/2404.02138v4)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: The indistinguishability of Large Language Model (LLM) output from human-authored content poses significant challenges, raising concerns about potential misuse of AI-generated text and its influence on future AI model training. Watermarking algorithms offer a viable solution by embedding detectable signatures into generated text. However, existing watermarking methods often entail trade-offs among attack robustness, generation quality, and additional overhead such as specialized frameworks or complex integrations. We propose a lightweight, topic-guided watermarking scheme for LLMs that partitions the vocabulary into topic-aligned token subsets. Given an input prompt, the scheme selects a relevant topic-specific token list, effectively "green-listing" semantically aligned tokens to embed robust marks while preserving the text's fluency and coherence. Experimental results across multiple LLMs and state-of-the-art benchmarks demonstrate that our method achieves comparable perplexity to industry-leading systems, including Google's SynthID-Text, yet enhances watermark robustness against paraphrasing and lexical perturbation attacks while introducing minimal performance overhead. Our approach avoids reliance on additional mechanisms beyond standard text generation pipelines, facilitating straightforward adoption, suggesting a practical path toward globally consistent watermarking of AI-generated content.



## **44. Watermarking Low-entropy Generation for Large Language Models: An Unbiased and Low-risk Method**

cs.CL

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2405.14604v3) [paper-pdf](http://arxiv.org/pdf/2405.14604v3)

**Authors**: Minjia Mao, Dongjun Wei, Zeyu Chen, Xiao Fang, Michael Chau

**Abstract**: Recent advancements in large language models (LLMs) have highlighted the risk of misusing them, raising the need for accurate detection of LLM-generated content. In response, a viable solution is to inject imperceptible identifiers into LLMs, known as watermarks. Our research extends the existing watermarking methods by proposing the novel Sampling One Then Accepting (STA-1) method. STA-1 is an unbiased watermark that preserves the original token distribution in expectation and has a lower risk of producing unsatisfactory outputs in low-entropy scenarios compared to existing unbiased watermarks. In watermark detection, STA-1 does not require prompts or a white-box LLM, provides statistical guarantees, demonstrates high efficiency in detection time, and remains robust against various watermarking attacks. Experimental results on low-entropy and high-entropy datasets demonstrate that STA-1 achieves the above properties simultaneously, making it a desirable solution for watermarking LLMs. Implementation codes for this study are available online.



## **45. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

cs.CL

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.07163v3) [paper-pdf](http://arxiv.org/pdf/2410.07163v3)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: This work studies the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences (e.g., copyrighted or harmful content) while preserving model utility. Despite the increasing demand for unlearning, a technically-grounded optimization framework is lacking. Gradient ascent (GA)-type methods, though widely used, are suboptimal as they reverse the learning process without controlling optimization divergence (i.e., deviation from the pre-trained state), leading to risks of over-forgetting and potential model collapse. Negative preference optimization (NPO) has been proposed to address this issue and is considered one of the state-of-the-art LLM unlearning approaches. In this work, we revisit NPO and identify another critical issue: reference model bias. This bias arises from using the reference model (i.e., the model prior to unlearning) to evaluate the unlearning success, which can compromise NPO's effectiveness. Specifically, it leads to (a) uneven allocation of optimization power across forget data with varying difficulty levels and (b) ineffective gradient weight smoothing during the early stages of unlearning optimization. To overcome these challenges, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that `simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We provide deeper insights into SimNPO's advantages through an analysis based on mixtures of Markov chains. Extensive experiments further validate SimNPO's efficacy on benchmarks like TOFU and MUSE, as well as its robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.



## **46. Do Unlearning Methods Remove Information from Language Model Weights?**

cs.LG

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.08827v3) [paper-pdf](http://arxiv.org/pdf/2410.08827v3)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods for information learned during pretraining, revealing the limitations of these methods in removing information from the model weights. Our results also suggest that unlearning evaluations that measure unlearning robustness on information learned during an additional fine-tuning phase may overestimate robustness compared to evaluations that attempt to unlearn information learned during pretraining.



## **47. GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs**

cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2411.13757v2) [paper-pdf](http://arxiv.org/pdf/2411.13757v2)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.



## **48. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection**

cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2412.10198v2) [paper-pdf](http://arxiv.org/pdf/2412.10198v2)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.



## **49. Enhancing Phishing Email Identification with Large Language Models**

cs.CR

9 pages, 5 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04759v1) [paper-pdf](http://arxiv.org/pdf/2502.04759v1)

**Authors**: Catherine Lee

**Abstract**: Phishing has long been a common tactic used by cybercriminals and continues to pose a significant threat in today's digital world. When phishing attacks become more advanced and sophisticated, there is an increasing need for effective methods to detect and prevent them. To address the challenging problem of detecting phishing emails, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. In this work, we take steps to study the efficacy of large language models (LLMs) in detecting phishing emails. The experiments show that the LLM achieves a high accuracy rate at high precision; importantly, it also provides interpretable evidence for the decisions.



## **50. Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models**

cs.CR

Accepted by ICLR2025. url: https://openreview.net/forum?id=s20W12XTF8

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.02298v4) [paper-pdf](http://arxiv.org/pdf/2410.02298v4)

**Authors**: Guobin Shen, Dongcheng Zhao, Yiting Dong, Xiang He, Yi Zeng

**Abstract**: As large language models (LLMs) become integral to various applications, ensuring both their safety and utility is paramount. Jailbreak attacks, which manipulate LLMs into generating harmful content, pose significant challenges to this balance. Existing defenses, such as prompt engineering and safety fine-tuning, often introduce computational overhead, increase inference latency, and lack runtime flexibility. Moreover, overly restrictive safety measures can degrade model utility by causing refusals of benign queries. In this paper, we introduce Jailbreak Antidote, a method that enables real-time adjustment of LLM safety preferences by manipulating a sparse subset of the model's internal states during inference. By shifting the model's hidden representations along a safety direction with varying strengths, we achieve flexible control over the safety-utility balance without additional token overhead or inference delays. Our analysis reveals that safety-related information in LLMs is sparsely distributed; adjusting approximately 5% of the internal state is as effective as modifying the entire state. Extensive experiments on nine LLMs (ranging from 2 billion to 72 billion parameters), evaluated against ten jailbreak attack methods and compared with six defense strategies, validate the effectiveness and efficiency of our approach. By directly manipulating internal states during reasoning, Jailbreak Antidote offers a lightweight, scalable solution that enhances LLM safety while preserving utility, opening new possibilities for real-time safety mechanisms in widely-deployed AI systems.



