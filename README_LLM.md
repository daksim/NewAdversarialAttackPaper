# Latest Large Language Model Attack Papers
**update at 2024-12-18 10:29:20**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Truthful Text Sanitization Guided by Inference Attacks**

cs.CL

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12928v1) [paper-pdf](http://arxiv.org/pdf/2412.12928v1)

**Authors**: Ildikó Pilán, Benet Manzanares-Salor, David Sánchez, Pierre Lison

**Abstract**: The purpose of text sanitization is to rewrite those text spans in a document that may directly or indirectly identify an individual, to ensure they no longer disclose personal information. Text sanitization must strike a balance between preventing the leakage of personal information (privacy protection) while also retaining as much of the document's original content as possible (utility preservation). We present an automated text sanitization strategy based on generalizations, which are more abstract (but still informative) terms that subsume the semantic content of the original text spans. The approach relies on instruction-tuned large language models (LLMs) and is divided into two stages. The LLM is first applied to obtain truth-preserving replacement candidates and rank them according to their abstraction level. Those candidates are then evaluated for their ability to protect privacy by conducting inference attacks with the LLM. Finally, the system selects the most informative replacement shown to be resistant to those attacks. As a consequence of this two-stage process, the chosen replacements effectively balance utility and privacy. We also present novel metrics to automatically evaluate these two aspects without the need to manually annotate data. Empirical results on the Text Anonymization Benchmark show that the proposed approach leads to enhanced utility, with only a marginal increase in the risk of re-identifying protected individuals compared to fully suppressing the original information. Furthermore, the selected replacements are shown to be more truth-preserving and abstractive than previous methods.



## **2. PROSAC: Provably Safe Certification for Machine Learning Models under Adversarial Attacks**

cs.LG

Accepted to AAAI2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2402.02629v2) [paper-pdf](http://arxiv.org/pdf/2402.02629v2)

**Authors**: Chen Feng, Ziquan Liu, Zhuo Zhi, Ilija Bogunovic, Carsten Gerner-Beuerle, Miguel Rodrigues

**Abstract**: It is widely known that state-of-the-art machine learning models, including vision and language models, can be seriously compromised by adversarial perturbations. It is therefore increasingly relevant to develop capabilities to certify their performance in the presence of the most effective adversarial attacks. Our paper offers a new approach to certify the performance of machine learning models in the presence of adversarial attacks with population level risk guarantees. In particular, we introduce the notion of $(\alpha,\zeta)$-safe machine learning model. We propose a hypothesis testing procedure, based on the availability of a calibration set, to derive statistical guarantees providing that the probability of declaring that the adversarial (population) risk of a machine learning model is less than $\alpha$ (i.e. the model is safe), while the model is in fact unsafe (i.e. the model adversarial population risk is higher than $\alpha$), is less than $\zeta$. We also propose Bayesian optimization algorithms to determine efficiently whether a machine learning model is $(\alpha,\zeta)$-safe in the presence of an adversarial attack, along with statistical guarantees. We apply our framework to a range of machine learning models - including various sizes of vision Transformer (ViT) and ResNet models - impaired by a variety of adversarial attacks, such as PGDAttack, MomentumAttack, GenAttack and BanditAttack, to illustrate the operation of our approach. Importantly, we show that ViT's are generally more robust to adversarial attacks than ResNets, and large models are generally more robust than smaller models. Our approach goes beyond existing empirical adversarial risk-based certification guarantees. It formulates rigorous (and provable) performance guarantees that can be used to satisfy regulatory requirements mandating the use of state-of-the-art technical tools.



## **3. RemoteRAG: A Privacy-Preserving LLM Cloud RAG Service**

cs.IR

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12775v1) [paper-pdf](http://arxiv.org/pdf/2412.12775v1)

**Authors**: Yihang Cheng, Lan Zhang, Junyang Wang, Mu Yuan, Yunhao Yao

**Abstract**: Retrieval-augmented generation (RAG) improves the service quality of large language models by retrieving relevant documents from credible literature and integrating them into the context of the user query. Recently, the rise of the cloud RAG service has made it possible for users to query relevant documents conveniently. However, directly sending queries to the cloud brings potential privacy leakage. In this paper, we are the first to formally define the privacy-preserving cloud RAG service to protect the user query and propose RemoteRAG as a solution regarding privacy, efficiency, and accuracy. For privacy, we introduce $(n,\epsilon)$-DistanceDP to characterize privacy leakage of the user query and the leakage inferred from relevant documents. For efficiency, we limit the search range from the total documents to a small number of selected documents related to a perturbed embedding generated from $(n,\epsilon)$-DistanceDP, so that computation and communication costs required for privacy protection significantly decrease. For accuracy, we ensure that the small range includes target documents related to the user query with detailed theoretical analysis. Experimental results also demonstrate that RemoteRAG can resist existing embedding inversion attack methods while achieving no loss in retrieval under various settings. Moreover, RemoteRAG is efficient, incurring only $0.67$ seconds and $46.66$KB of data transmission ($2.72$ hours and $1.43$ GB with the non-optimized privacy-preserving scheme) when retrieving from a total of $10^6$ documents.



## **4. Defending LVLMs Against Vision Attacks through Partial-Perception Supervision**

cs.CV

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12722v1) [paper-pdf](http://arxiv.org/pdf/2412.12722v1)

**Authors**: Qi Zhou, Tianlin Li, Qing Guo, Dongxia Wang, Yun Lin, Yang Liu, Jin Song Dong

**Abstract**: Recent studies have raised significant concerns regarding the vulnerability of Large Vision Language Models (LVLMs) to maliciously injected or perturbed input images, which can mislead their responses. Existing defense methods show that such vision attacks are sensitive to image modifications especially cropping, using majority voting across responses of modified images as corrected responses. However, these modifications often result in partial images and distort the semantics, which reduces response quality on clean images after voting. Instead of directly using responses from partial images for voting, we investigate using them to supervise the LVLM's responses to the original images. We propose a black-box, training-free method called DPS (Defense through Partial-Perception Supervision). In this approach, the model is prompted using the responses generated by a model that perceives only a partial image. With DPS, the model can adjust its response based on partial image understanding when under attack, while confidently maintaining its original response for clean input. Our findings show that the weak model can supervise the strong model: when faced with an attacked input, the strong model becomes less confident and adjusts its response based on the weak model's partial understanding, effectively defending against the attack. With clean input, it confidently maintains its original response. Empirical experiments show our method outperforms the baseline, cutting the average attack success rate by 76.3% across six datasets on three popular models.



## **5. Jailbreaking? One Step Is Enough!**

cs.CL

17 pages

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12621v1) [paper-pdf](http://arxiv.org/pdf/2412.12621v1)

**Authors**: Weixiong Zheng, Peijian Zeng, Yiwei Li, Hongyan Wu, Nankai Lin, Junhao Chen, Aimin Yang, Yongmei Zhou

**Abstract**: Large language models (LLMs) excel in various tasks but remain vulnerable to jailbreak attacks, where adversaries manipulate prompts to generate harmful outputs. Examining jailbreak prompts helps uncover the shortcomings of LLMs. However, current jailbreak methods and the target model's defenses are engaged in an independent and adversarial process, resulting in the need for frequent attack iterations and redesigning attacks for different models. To address these gaps, we propose a Reverse Embedded Defense Attack (REDA) mechanism that disguises the attack intention as the "defense". intention against harmful content. Specifically, REDA starts from the target response, guiding the model to embed harmful content within its defensive measures, thereby relegating harmful content to a secondary role and making the model believe it is performing a defensive task. The attacking model considers that it is guiding the target model to deal with harmful content, while the target model thinks it is performing a defensive task, creating an illusion of cooperation between the two. Additionally, to enhance the model's confidence and guidance in "defensive" intentions, we adopt in-context learning (ICL) with a small number of attack examples and construct a corresponding dataset of attack examples. Extensive evaluations demonstrate that the REDA method enables cross-model attacks without the need to redesign attack strategies for different models, enables successful jailbreak in one iteration, and outperforms existing methods on both open-source and closed-source models.



## **6. Jailbreak Large Vision-Language Models Through Multi-Modal Linkage**

cs.CV

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.00473v4) [paper-pdf](http://arxiv.org/pdf/2412.00473v4)

**Authors**: Yu Wang, Xiaofei Zhou, Yichen Wang, Geyuan Zhang, Tianxing He

**Abstract**: With the significant advancement of Large Vision-Language Models (VLMs), concerns about their potential misuse and abuse have grown rapidly. Previous studies have highlighted VLMs' vulnerability to jailbreak attacks, where carefully crafted inputs can lead the model to produce content that violates ethical and legal standards. However, existing methods struggle against state-of-the-art VLMs like GPT-4o, due to the over-exposure of harmful content and lack of stealthy malicious guidance. In this work, we propose a novel jailbreak attack framework: Multi-Modal Linkage (MML) Attack. Drawing inspiration from cryptography, MML utilizes an encryption-decryption process across text and image modalities to mitigate over-exposure of malicious information. To align the model's output with malicious intent covertly, MML employs a technique called "evil alignment", framing the attack within a video game production scenario. Comprehensive experiments demonstrate MML's effectiveness. Specifically, MML jailbreaks GPT-4o with attack success rates of 97.80% on SafeBench, 98.81% on MM-SafeBench and 99.07% on HADES-Dataset. Our code is available at https://github.com/wangyu-ovo/MML



## **7. Task-Agnostic Language Model Watermarking via High Entropy Passthrough Layers**

cs.CL

Accepted to AAAI2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12563v1) [paper-pdf](http://arxiv.org/pdf/2412.12563v1)

**Authors**: Vaden Masrani, Mohammad Akbari, David Ming Xuan Yue, Ahmad Rezaei, Yong Zhang

**Abstract**: In the era of costly pre-training of large language models, ensuring the intellectual property rights of model owners, and insuring that said models are responsibly deployed, is becoming increasingly important. To this end, we propose model watermarking via passthrough layers, which are added to existing pre-trained networks and trained using a self-supervised loss such that the model produces high-entropy output when prompted with a unique private key, and acts normally otherwise. Unlike existing model watermarking methods, our method is fully task-agnostic, and can be applied to both classification and sequence-to-sequence tasks without requiring advanced access to downstream fine-tuning datasets. We evaluate the proposed passthrough layers on a wide range of downstream tasks, and show experimentally our watermarking method achieves a near-perfect watermark extraction accuracy and false-positive rate in most cases without damaging original model performance. Additionally, we show our method is robust to both downstream fine-tuning, fine-pruning, and layer removal attacks, and can be trained in a fraction of the time required to train the original model. Code is available in the paper.



## **8. Recent advancements in LLM Red-Teaming: Techniques, Defenses, and Ethical Considerations**

cs.CL

16 pages, 2 figures

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2410.09097v2) [paper-pdf](http://arxiv.org/pdf/2410.09097v2)

**Authors**: Tarun Raheja, Nilay Pochhi, F. D. C. M. Curie

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing tasks, but their vulnerability to jailbreak attacks poses significant security risks. This survey paper presents a comprehensive analysis of recent advancements in attack strategies and defense mechanisms within the field of Large Language Model (LLM) red-teaming. We analyze various attack methods, including gradient-based optimization, reinforcement learning, and prompt engineering approaches. We discuss the implications of these attacks on LLM safety and the need for improved defense mechanisms. This work aims to provide a thorough understanding of the current landscape of red-teaming attacks and defenses on LLMs, enabling the development of more secure and reliable language models.



## **9. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

cs.LG

accepted by KDD2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2408.08685v2) [paper-pdf](http://arxiv.org/pdf/2408.08685v2)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.



## **10. NLSR: Neuron-Level Safety Realignment of Large Language Models Against Harmful Fine-Tuning**

cs.CL

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12497v1) [paper-pdf](http://arxiv.org/pdf/2412.12497v1)

**Authors**: Xin Yi, Shunfan Zheng, Linlin Wang, Gerard de Melo, Xiaoling Wang, Liang He

**Abstract**: The emergence of finetuning-as-a-service has revealed a new vulnerability in large language models (LLMs). A mere handful of malicious data uploaded by users can subtly manipulate the finetuning process, resulting in an alignment-broken model. Existing methods to counteract fine-tuning attacks typically require substantial computational resources. Even with parameter-efficient techniques like LoRA, gradient updates remain essential. To address these challenges, we propose \textbf{N}euron-\textbf{L}evel \textbf{S}afety \textbf{R}ealignment (\textbf{NLSR}), a training-free framework that restores the safety of LLMs based on the similarity difference of safety-critical neurons before and after fine-tuning. The core of our framework is first to construct a safety reference model from an initially aligned model to amplify safety-related features in neurons. We then utilize this reference model to identify safety-critical neurons, which we prepare as patches. Finally, we selectively restore only those neurons that exhibit significant similarity differences by transplanting these prepared patches, thereby minimally altering the fine-tuned model. Extensive experiments demonstrate significant safety enhancements in fine-tuned models across multiple downstream tasks, while greatly maintaining task-level accuracy. Our findings suggest regions of some safety-critical neurons show noticeable differences after fine-tuning, which can be effectively corrected by transplanting neurons from the reference model without requiring additional training. The code will be available at \url{https://github.com/xinykou/NLSR}



## **11. Adversarial Attacks on Large Language Models in Medicine**

cs.AI

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2406.12259v3) [paper-pdf](http://arxiv.org/pdf/2406.12259v3)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.



## **12. When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations**

cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2411.12701v2) [paper-pdf](http://arxiv.org/pdf/2411.12701v2)

**Authors**: Huaizhi Ge, Yiming Li, Qifan Wang, Yongfeng Zhang, Ruixiang Tang

**Abstract**: Large Language Models (LLMs) are known to be vulnerable to backdoor attacks, where triggers embedded in poisoned samples can maliciously alter LLMs' behaviors. In this paper, we move beyond attacking LLMs and instead examine backdoor attacks through the novel lens of natural language explanations. Specifically, we leverage LLMs' generative capabilities to produce human-readable explanations for their decisions, enabling direct comparisons between explanations for clean and poisoned samples. Our results show that backdoored models produce coherent explanations for clean inputs but diverse and logically flawed explanations for poisoned data, a pattern consistent across classification and generation tasks for different backdoor attacks. Further analysis reveals key insights into the explanation generation process. At the token level, explanation tokens associated with poisoned samples only appear in the final few transformer layers. At the sentence level, attention dynamics indicate that poisoned inputs shift attention away from the original input context during explanation generation. These findings enhance our understanding of backdoor mechanisms in LLMs and present a promising framework for detecting vulnerabilities through explainability.



## **13. Stepwise Reasoning Error Disruption Attack of LLMs**

cs.AI

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11934v1) [paper-pdf](http://arxiv.org/pdf/2412.11934v1)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications.



## **14. PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization**

cs.CR

Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for  Toxicity Maximization

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.05892v2) [paper-pdf](http://arxiv.org/pdf/2412.05892v2)

**Authors**: Ruoxi Cheng, Yizhong Ding, Shuirong Cao, Ranjie Duan, Xiaoshuang Jia, Shaowei Yuan, Zhiqiang Wang, Xiaojun Jia

**Abstract**: Understanding the vulnerabilities of Large Vision Language Models (LVLMs) to jailbreak attacks is essential for their responsible real-world deployment. Most previous work requires access to model gradients, or is based on human knowledge (prompt engineering) to complete jailbreak, and they hardly consider the interaction of images and text, resulting in inability to jailbreak in black box scenarios or poor performance. To overcome these limitations, we propose a Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for toxicity maximization, referred to as PBI-Attack. Our method begins by extracting malicious features from a harmful corpus using an alternative LVLM and embedding these features into a benign image as prior information. Subsequently, we enhance these features through bidirectional cross-modal interaction optimization, which iteratively optimizes the bimodal perturbations in an alternating manner through greedy search, aiming to maximize the toxicity of the generated response. The toxicity level is quantified using a well-trained evaluation model. Experiments demonstrate that PBI-Attack outperforms previous state-of-the-art jailbreak methods, achieving an average attack success rate of 92.5% across three open-source LVLMs and around 67.3% on three closed-source LVLMs. Disclaimer: This paper contains potentially disturbing and offensive content.



## **15. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2408.11749v2) [paper-pdf](http://arxiv.org/pdf/2408.11749v2)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.



## **16. Intention Analysis Makes LLMs A Good Jailbreak Defender**

cs.CL

COLING 2025

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2401.06561v4) [paper-pdf](http://arxiv.org/pdf/2401.06561v4)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly when facing complex and stealthy jailbreak attacks, presents a formidable challenge. Unfortunately, existing methods often overlook this intrinsic nature of jailbreaks, which limits their effectiveness in such complex scenarios. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis ($\mathbb{IA}$). $\mathbb{IA}$ works by triggering LLMs' inherent self-correct and improve ability through a two-stage process: 1) analyzing the essential intention of the user input, and 2) providing final policy-aligned responses based on the first round conversation. Notably, $\mathbb{IA}$ is an inference-only method, thus could enhance LLM safety without compromising their helpfulness. Extensive experiments on varying jailbreak benchmarks across a wide range of LLMs show that $\mathbb{IA}$ could consistently and significantly reduce the harmfulness in responses (averagely -48.2% attack success rate). Encouragingly, with our $\mathbb{IA}$, Vicuna-7B even outperforms GPT-3.5 regarding attack success rate. We empirically demonstrate that, to some extent, $\mathbb{IA}$ is robust to errors in generated intentions. Further analyses reveal the underlying principle of $\mathbb{IA}$: suppressing LLM's tendency to follow jailbreak prompts, thereby enhancing safety.



## **17. Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift**

cs.CV

11 pages, 9 figures

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2406.18844v4) [paper-pdf](http://arxiv.org/pdf/2406.18844v4)

**Authors**: Siyuan Liang, Jiawei Liang, Tianyu Pang, Chao Du, Aishan Liu, Mingli Zhu, Xiaochun Cao, Dacheng Tao

**Abstract**: Instruction tuning enhances large vision-language models (LVLMs) but increases their vulnerability to backdoor attacks due to their open design. Unlike prior studies in static settings, this paper explores backdoor attacks in LVLM instruction tuning across mismatched training and testing domains. We introduce a new evaluation dimension, backdoor domain generalization, to assess attack robustness under visual and text domain shifts. Our findings reveal two insights: (1) backdoor generalizability improves when distinctive trigger patterns are independent of specific data domains or model architectures, and (2) the competitive interaction between trigger patterns and clean semantic regions, where guiding the model to predict triggers enhances attack generalizability. Based on these insights, we propose a multimodal attribution backdoor attack (MABA) that injects domain-agnostic triggers into critical areas using attributional interpretation. Experiments with OpenFlamingo, Blip-2, and Otter show that MABA significantly boosts the attack success rate of generalization by 36.4%, achieving a 97% success rate at a 0.2% poisoning rate. This study reveals limitations in current evaluations and highlights how enhanced backdoor generalizability poses a security threat to LVLMs, even without test data access.



## **18. Exploiting the Index Gradients for Optimization-Based Jailbreaking on Large Language Models**

cs.CL

13 pages,2 figures, accepted by COLING 2025

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.08615v2) [paper-pdf](http://arxiv.org/pdf/2412.08615v2)

**Authors**: Jiahui Li, Yongchang Hao, Haoyu Xu, Xing Wang, Yu Hong

**Abstract**: Despite the advancements in training Large Language Models (LLMs) with alignment techniques to enhance the safety of generated content, these models remain susceptible to jailbreak, an adversarial attack method that exposes security vulnerabilities in LLMs. Notably, the Greedy Coordinate Gradient (GCG) method has demonstrated the ability to automatically generate adversarial suffixes that jailbreak state-of-the-art LLMs. However, the optimization process involved in GCG is highly time-consuming, rendering the jailbreaking pipeline inefficient. In this paper, we investigate the process of GCG and identify an issue of Indirect Effect, the key bottleneck of the GCG optimization. To this end, we propose the Model Attack Gradient Index GCG (MAGIC), that addresses the Indirect Effect by exploiting the gradient information of the suffix tokens, thereby accelerating the procedure by having less computation and fewer iterations. Our experiments on AdvBench show that MAGIC achieves up to a 1.5x speedup, while maintaining Attack Success Rates (ASR) on par or even higher than other baselines. Our MAGIC achieved an ASR of 74% on the Llama-2 and an ASR of 54% when conducting transfer attacks on GPT-3.5. Code is available at https://github.com/jiah-li/magic.



## **19. Failures to Find Transferable Image Jailbreaks Between Vision-Language Models**

cs.CL

NeurIPS 2024 Workshops: RBFM (Best Paper), Frontiers in AdvML (Oral),  Red Teaming GenAI (Oral), SoLaR (Spotlight), SATA

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2407.15211v2) [paper-pdf](http://arxiv.org/pdf/2407.15211v2)

**Authors**: Rylan Schaeffer, Dan Valentine, Luke Bailey, James Chua, Cristóbal Eyzaguirre, Zane Durante, Joe Benton, Brando Miranda, Henry Sleight, John Hughes, Rajashree Agrawal, Mrinank Sharma, Scott Emmons, Sanmi Koyejo, Ethan Perez

**Abstract**: The integration of new modalities into frontier AI systems offers exciting capabilities, but also increases the possibility such systems can be adversarially manipulated in undesirable ways. In this work, we focus on a popular class of vision-language models (VLMs) that generate text outputs conditioned on visual and textual inputs. We conducted a large-scale empirical study to assess the transferability of gradient-based universal image ``jailbreaks" using a diverse set of over 40 open-parameter VLMs, including 18 new VLMs that we publicly release. Overall, we find that transferable gradient-based image jailbreaks are extremely difficult to obtain. When an image jailbreak is optimized against a single VLM or against an ensemble of VLMs, the jailbreak successfully jailbreaks the attacked VLM(s), but exhibits little-to-no transfer to any other VLMs; transfer is not affected by whether the attacked and target VLMs possess matching vision backbones or language models, whether the language model underwent instruction-following and/or safety-alignment training, or many other factors. Only two settings display partially successful transfer: between identically-pretrained and identically-initialized VLMs with slightly different VLM training data, and between different training checkpoints of a single VLM. Leveraging these results, we then demonstrate that transfer can be significantly improved against a specific target VLM by attacking larger ensembles of ``highly-similar" VLMs. These results stand in stark contrast to existing evidence of universal and transferable text jailbreaks against language models and transferable adversarial attacks against image classifiers, suggesting that VLMs may be more robust to gradient-based transfer attacks.



## **20. Finding a Wolf in Sheep's Clothing: Combating Adversarial Text-To-Image Prompts with Text Summarization**

cs.CR

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.12212v1) [paper-pdf](http://arxiv.org/pdf/2412.12212v1)

**Authors**: Portia Cooper, Harshita Narnoli, Mihai Surdeanu

**Abstract**: Text-to-image models are vulnerable to the stepwise "Divide-and-Conquer Attack" (DACA) that utilize a large language model to obfuscate inappropriate content in prompts by wrapping sensitive text in a benign narrative. To mitigate stepwise DACA attacks, we propose a two-layer method involving text summarization followed by binary classification. We assembled the Adversarial Text-to-Image Prompt (ATTIP) dataset ($N=940$), which contained DACA-obfuscated and non-obfuscated prompts. From the ATTIP dataset, we created two summarized versions: one generated by a small encoder model and the other by a large language model. Then, we used an encoder classifier and a GPT-4o classifier to perform content moderation on the summarized and unsummarized prompts. When compared with a classifier that operated over the unsummarized data, our method improved F1 score performance by 31%. Further, the highest recorded F1 score achieved (98%) was produced by the encoder classifier on a summarized ATTIP variant. This study indicates that pre-classification text summarization can inoculate content detection models against stepwise DACA obfuscations.



## **21. Red Teaming GPT-4V: Are GPT-4V Safe Against Uni/Multi-Modal Jailbreak Attacks?**

cs.LG

technical report; update code repo link

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2404.03411v2) [paper-pdf](http://arxiv.org/pdf/2404.03411v2)

**Authors**: Shuo Chen, Zhen Han, Bailan He, Zifeng Ding, Wenqian Yu, Philip Torr, Volker Tresp, Jindong Gu

**Abstract**: Various jailbreak attacks have been proposed to red-team Large Language Models (LLMs) and revealed the vulnerable safeguards of LLMs. Besides, some methods are not limited to the textual modality and extend the jailbreak attack to Multimodal Large Language Models (MLLMs) by perturbing the visual input. However, the absence of a universal evaluation benchmark complicates the performance reproduction and fair comparison. Besides, there is a lack of comprehensive evaluation of closed-source state-of-the-art (SOTA) models, especially MLLMs, such as GPT-4V. To address these issues, this work first builds a comprehensive jailbreak evaluation dataset with 1445 harmful questions covering 11 different safety policies. Based on this dataset, extensive red-teaming experiments are conducted on 11 different LLMs and MLLMs, including both SOTA proprietary models and open-source models. We then conduct a deep analysis of the evaluated results and find that (1) GPT4 and GPT-4V demonstrate better robustness against jailbreak attacks compared to open-source LLMs and MLLMs. (2) Llama2 and Qwen-VL-Chat are more robust compared to other open-source models. (3) The transferability of visual jailbreak methods is relatively limited compared to textual jailbreak methods. The dataset and code can be found https://github.com/chenxshuo/RedTeamingGPT4V



## **22. The Superalignment of Superhuman Intelligence with Large Language Models**

cs.CL

Under review of Science China

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11145v1) [paper-pdf](http://arxiv.org/pdf/2412.11145v1)

**Authors**: Minlie Huang, Yingkang Wang, Shiyao Cui, Pei Ke, Jie Tang

**Abstract**: We have witnessed superhuman intelligence thanks to the fast development of large language models and multimodal language models. As the application of such superhuman models becomes more and more common, a critical question rises here: how can we ensure superhuman models are still safe, reliable and aligned well to human values? In this position paper, we discuss the concept of superalignment from the learning perspective to answer this question by outlining the learning paradigm shift from large-scale pretraining, supervised fine-tuning, to alignment training. We define superalignment as designing effective and efficient alignment algorithms to learn from noisy-labeled data (point-wise samples or pair-wise preference data) in a scalable way when the task becomes very complex for human experts to annotate and the model is stronger than human experts. We highlight some key research problems in superalignment, namely, weak-to-strong generalization, scalable oversight, and evaluation. We then present a conceptual framework for superalignment, which consists of three modules: an attacker which generates adversary queries trying to expose the weaknesses of a learner model; a learner which will refine itself by learning from scalable feedbacks generated by a critic model along with minimal human experts; and a critic which generates critics or explanations for a given query-response pair, with a target of improving the learner by criticizing. We discuss some important research problems in each component of this framework and highlight some interesting research ideas that are closely related to our proposed framework, for instance, self-alignment, self-play, self-refinement, and more. Last, we highlight some future research directions for superalignment, including identification of new emergent risks and multi-dimensional alignment.



## **23. Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models**

cs.CV

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2404.10335v4) [paper-pdf](http://arxiv.org/pdf/2404.10335v4)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo

**Abstract**: Adversarial attacks, particularly \textbf{targeted} transfer-based attacks, can be used to assess the adversarial robustness of large visual-language models (VLMs), allowing for a more thorough examination of potential security flaws before deployment. However, previous transfer-based adversarial attacks incur high costs due to high iteration counts and complex method structure. Furthermore, due to the unnaturalness of adversarial semantics, the generated adversarial examples have low transferability. These issues limit the utility of existing methods for assessing robustness. To address these issues, we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask method to disperse adversarial semantics throughout the image rather than concentrating them in a single area. Finally, AdvDiffVLM embeds more target semantics into adversarial examples after multiple iterations. Experimental results show that our method generates adversarial examples 5x to 10x faster than state-of-the-art transfer-based adversarial attacks while maintaining higher quality adversarial examples. Furthermore, compared to previous transfer-based adversarial attacks, the adversarial examples generated by our method have better transferability. Notably, AdvDiffVLM can successfully attack a variety of commercial VLMs in a black-box environment, including GPT-4V.



## **24. Do Chase Your Tail! Missing Key Aspects Augmentation in Textual Vulnerability Descriptions of Long-tail Software through Feature Inference**

cs.SE

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2405.07430v2) [paper-pdf](http://arxiv.org/pdf/2405.07430v2)

**Authors**: Linyi Han, Shidong Pan, Zhenchang Xing, Jiamou Sun, Sofonias Yitagesu, Xiaowang Zhang, Zhiyong Feng

**Abstract**: Augmenting missing key aspects in Textual Vulnerability Descriptions (TVDs) is crucial for effective vulnerability analysis. For instance, in TVDs, key aspects include Attack Vector, Vulnerability Type, among others. These key aspects help security engineers understand and address the vulnerability in a timely manner. For software with a large user base (non-long-tail software), augmenting these missing key aspects has significantly advanced vulnerability analysis and software security research. However, software instances with a limited user base (long-tail software) often get overlooked due to inconsistency software names, TVD limited avaliability, and domain-specific jargon, which complicates vulnerability analysis and software repairs. In this paper, we introduce a novel software feature inference framework designed to augment the missing key aspects of TVDs for long-tail software. Firstly, we tackle the issue of non-standard software names found in community-maintained vulnerability databases by cross-referencing government databases with Common Vulnerabilities and Exposures (CVEs). Next, we employ Large Language Models (LLMs) to generate the missing key aspects. However, the limited availability of historical TVDs restricts the variety of examples. To overcome this limitation, we utilize the Common Weakness Enumeration (CWE) to classify all TVDs and select cluster centers as representative examples. To ensure accuracy, we present Natural Language Inference (NLI) models specifically designed for long-tail software. These models identify and eliminate incorrect responses. Additionally, we use a wiki repository to provide explanations for proprietary terms.



## **25. Simulate and Eliminate: Revoke Backdoors for Generative Large Language Models**

cs.CR

To appear at AAAI 2025

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2405.07667v2) [paper-pdf](http://arxiv.org/pdf/2405.07667v2)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive data. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle scenarios where trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike other works that assume access to cleanly trained models, our safety-enhanced LLMs are able to revoke backdoors without any reference. Consequently, our safety-enhanced LLMs no longer produce targeted responses when the backdoor triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability.



## **26. Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety Re-Alignment for Fine-Tuned Language Models**

cs.CL

14 pages, 12 figures,

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11041v1) [paper-pdf](http://arxiv.org/pdf/2412.11041v1)

**Authors**: Di Wu, Xin Lu, Yanyan Zhao, Bing Qin

**Abstract**: Although large language models (LLMs) achieve effective safety alignment at the time of release, they still face various safety challenges. A key issue is that fine-tuning often compromises the safety alignment of LLMs. To address this issue, we propose a method named \textbf{IRR} (\textbf{I}dentify, \textbf{R}emove, and \textbf{R}ecalibrate for Safety Realignment) that performs safety realignment for LLMs. The core of IRR is to identify and remove unsafe delta parameters from the fine-tuned models, while recalibrating the retained ones. We evaluate the effectiveness of IRR across various datasets, including both full fine-tuning and LoRA methods. Our results demonstrate that IRR significantly enhances the safety performance of fine-tuned models on safety benchmarks, such as harmful queries and jailbreak attacks, while maintaining their performance on downstream tasks. The source code is available at: \url{https://anonymous.4open.science/r/IRR-BD4F}.



## **27. Labeling NIDS Rules with MITRE ATT&CK Techniques: Machine Learning vs. Large Language Models**

cs.CR

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2412.10978v1) [paper-pdf](http://arxiv.org/pdf/2412.10978v1)

**Authors**: Nir Daniel, Florian Klaus Kaiser, Shay Giladi, Sapir Sharabi, Raz Moyal, Shalev Shpolyansky, Andres Murillo, Aviad Elyashar, Rami Puzis

**Abstract**: Analysts in Security Operations Centers (SOCs) are often occupied with time-consuming investigations of alerts from Network Intrusion Detection Systems (NIDS). Many NIDS rules lack clear explanations and associations with attack techniques, complicating the alert triage and the generation of attack hypotheses. Large Language Models (LLMs) may be a promising technology to reduce the alert explainability gap by associating rules with attack techniques. In this paper, we investigate the ability of three prominent LLMs (ChatGPT, Claude, and Gemini) to reason about NIDS rules while labeling them with MITRE ATT&CK tactics and techniques. We discuss prompt design and present experiments performed with 973 Snort rules. Our results indicate that while LLMs provide explainable, scalable, and efficient initial mappings, traditional Machine Learning (ML) models consistently outperform them in accuracy, achieving higher precision, recall, and F1-scores. These results highlight the potential for hybrid LLM-ML approaches to enhance SOC operations and better address the evolving threat landscape.



## **28. CEKER: A Generalizable LLM Framework for Literature Analysis with a Case Study in Unikernel Security**

cs.CR

7 pages, 2 figures

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2412.10904v1) [paper-pdf](http://arxiv.org/pdf/2412.10904v1)

**Authors**: Alex Wollman, John Hastings

**Abstract**: Literature reviews are a critical component of formulating and justifying new research, but are a manual and often time-consuming process. This research introduces a novel, generalizable approach to literature analysis called CEKER which uses a three-step process to streamline the collection of literature, the extraction of key insights, and the summarized analysis of key trends and gaps. Leveraging Large Language Models (LLMs), this methodology represents a significant shift from traditional manual literature reviews, offering a scalable, flexible, and repeatable approach that can be applied across diverse research domains.   A case study on unikernel security illustrates CEKER's ability to generate novel insights validated against previous manual methods. CEKER's analysis highlighted reduced attack surface as the most prominent theme. Key security gaps included the absence of Address Space Layout Randomization, missing debugging tools, and limited entropy generation, all of which represent important challenges to unikernel security. The study also revealed a reliance on hypervisors as a potential attack vector and emphasized the need for dynamic security adjustments to address real-time threats.



## **29. Towards Action Hijacking of Large Language Model-based Agent**

cs.CR

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2412.10807v1) [paper-pdf](http://arxiv.org/pdf/2412.10807v1)

**Authors**: Yuyang Zhang, Kangjie Chen, Xudong Jiang, Yuxiang Sun, Run Wang, Lina Wang

**Abstract**: In the past few years, intelligent agents powered by large language models (LLMs) have achieved remarkable progress in performing complex tasks. These LLM-based agents receive queries as tasks and decompose them into various subtasks via the equipped LLMs to guide the action of external entities (\eg{}, tools, AI-agents) to answer the questions from users. Empowered by their exceptional capabilities of understanding and problem-solving, they are widely adopted in labor-intensive sectors including healthcare, finance, code completion, \etc{} At the same time, there are also concerns about the potential misuse of these agents, prompting the built-in safety guards from service providers. To circumvent the built-in guidelines, the prior studies proposed a multitude of attacks including memory poisoning, jailbreak, and prompt injection. These studies often fail to maintain effectiveness across safety filters employed by agents due to the restricted privileges and the harmful semantics in queries. In this paper, we introduce \Name, a novel hijacking attack to manipulate the action plans of black-box agent system. \Name first collects the action-aware memory through prompt theft from long-term memory. It then leverages the internal memory retrieval mechanism of the agent to provide an erroneous context. The huge gap between the latent spaces of the retriever and safety filters allows our method to bypass the detection easily. Extensive experimental results demonstrate the effectiveness of our apporach (\eg{}, 99.67\% ASR). Besides, our approach achieved an average bypass rate of 92.7\% for safety filters.



## **30. On Effects of Steering Latent Representation for Large Language Model Unlearning**

cs.CL

Accepted at AAAI-25 Main Technical Track

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2408.06223v2) [paper-pdf](http://arxiv.org/pdf/2408.06223v2)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. We investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. We show that RMU unlearned models are robust against adversarial jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU -- a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.



## **31. No Free Lunch for Defending Against Prefilling Attack by In-Context Learning**

cs.CR

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.12192v1) [paper-pdf](http://arxiv.org/pdf/2412.12192v1)

**Authors**: Zhiyu Xue, Guangliang Liu, Bocheng Chen, Kristen Marie Johnson, Ramtin Pedarsani

**Abstract**: The security of Large Language Models (LLMs) has become an important research topic since the emergence of ChatGPT. Though there have been various effective methods to defend against jailbreak attacks, prefilling attacks remain an unsolved and popular threat against open-sourced LLMs. In-Context Learning (ICL) offers a computationally efficient defense against various jailbreak attacks, yet no effective ICL methods have been developed to counter prefilling attacks. In this paper, we: (1) show that ICL can effectively defend against prefilling jailbreak attacks by employing adversative sentence structures within demonstrations; (2) characterize the effectiveness of this defense through the lens of model size, number of demonstrations, over-defense, integration with other jailbreak attacks, and the presence of safety alignment. Given the experimental results and our analysis, we conclude that there is no free lunch for defending against prefilling jailbreak attacks with ICL. On the one hand, current safety alignment methods fail to mitigate prefilling jailbreak attacks, but adversative structures within ICL demonstrations provide robust defense across various model sizes and complex jailbreak attacks. On the other hand, LLMs exhibit similar over-defensiveness when utilizing ICL demonstrations with adversative structures, and this behavior appears to be independent of model size.



## **32. AdvPrefix: An Objective for Nuanced LLM Jailbreaks**

cs.LG

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.10321v1) [paper-pdf](http://arxiv.org/pdf/2412.10321v1)

**Authors**: Sicheng Zhu, Brandon Amos, Yuandong Tian, Chuan Guo, Ivan Evtimov

**Abstract**: Many jailbreak attacks on large language models (LLMs) rely on a common objective: making the model respond with the prefix "Sure, here is (harmful request)". While straightforward, this objective has two limitations: limited control over model behaviors, often resulting in incomplete or unrealistic responses, and a rigid format that hinders optimization. To address these limitations, we introduce AdvPrefix, a new prefix-forcing objective that enables more nuanced control over model behavior while being easy to optimize. Our objective leverages model-dependent prefixes, automatically selected based on two criteria: high prefilling attack success rates and low negative log-likelihood. It can further simplify optimization by using multiple prefixes for a single user request. AdvPrefix can integrate seamlessly into existing jailbreak attacks to improve their performance for free. For example, simply replacing GCG attack's target prefixes with ours on Llama-3 improves nuanced attack success rates from 14% to 80%, suggesting that current alignment struggles to generalize to unseen prefixes. Our work demonstrates the importance of jailbreak objectives in achieving nuanced jailbreaks.



## **33. RTL-Breaker: Assessing the Security of LLMs against Backdoor Attacks on HDL Code Generation**

cs.CR

Accepted at 2025 Design, Automation & Test in Europe (DATE)  Conference

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2411.17569v2) [paper-pdf](http://arxiv.org/pdf/2411.17569v2)

**Authors**: Lakshmi Likhitha Mankali, Jitendra Bhandari, Manaar Alam, Ramesh Karri, Michail Maniatakos, Ozgur Sinanoglu, Johann Knechtel

**Abstract**: Large language models (LLMs) have demonstrated remarkable potential with code generation/completion tasks for hardware design. In fact, LLM-based hardware description language (HDL) code generation has enabled the industry to realize complex designs more quickly, reducing the time and effort required in the development cycle. However, the increased reliance on such automation introduces critical security risks. Notably, given that LLMs have to be trained on vast datasets of codes that are typically sourced from publicly available repositories (often without thorough validation), LLMs are susceptible to so-called data poisoning or backdoor attacks. Here, attackers inject malicious code for the training data, which can be carried over into the HDL code generated by LLMs. This threat vector can compromise the security and integrity of entire hardware systems. In this work, we propose RTL-Breaker, a novel backdoor attack framework on LLM-based HDL code generation. RTL-Breaker provides an in-depth analysis for essential aspects of this novel problem: 1) various trigger mechanisms versus their effectiveness for inserting malicious modifications, and 2) side-effects by backdoor attacks on code generation in general, i.e., impact on code quality. RTL-Breaker emphasizes the urgent need for more robust measures to safeguard against such attacks. Toward that end, we open-source our framework and all data.



## **34. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection**

cs.CR

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.10198v1) [paper-pdf](http://arxiv.org/pdf/2412.10198v1)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.



## **35. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

cs.CL

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2411.07870v5) [paper-pdf](http://arxiv.org/pdf/2411.07870v5)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.



## **36. AdvWave: Stealthy Adversarial Jailbreak Attack against Large Audio-Language Models**

cs.SD

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08608v1) [paper-pdf](http://arxiv.org/pdf/2412.08608v1)

**Authors**: Mintong Kang, Chejian Xu, Bo Li

**Abstract**: Recent advancements in large audio-language models (LALMs) have enabled speech-based user interactions, significantly enhancing user experience and accelerating the deployment of LALMs in real-world applications. However, ensuring the safety of LALMs is crucial to prevent risky outputs that may raise societal concerns or violate AI regulations. Despite the importance of this issue, research on jailbreaking LALMs remains limited due to their recent emergence and the additional technical challenges they present compared to attacks on DNN-based audio models. Specifically, the audio encoders in LALMs, which involve discretization operations, often lead to gradient shattering, hindering the effectiveness of attacks relying on gradient-based optimizations. The behavioral variability of LALMs further complicates the identification of effective (adversarial) optimization targets. Moreover, enforcing stealthiness constraints on adversarial audio waveforms introduces a reduced, non-convex feasible solution space, further intensifying the challenges of the optimization process. To overcome these challenges, we develop AdvWave, the first jailbreak framework against LALMs. We propose a dual-phase optimization method that addresses gradient shattering, enabling effective end-to-end gradient-based optimization. Additionally, we develop an adaptive adversarial target search algorithm that dynamically adjusts the adversarial optimization target based on the response patterns of LALMs for specific queries. To ensure that adversarial audio remains perceptually natural to human listeners, we design a classifier-guided optimization approach that generates adversarial noise resembling common urban sounds. Extensive evaluations on multiple advanced LALMs demonstrate that AdvWave outperforms baseline methods, achieving a 40% higher average jailbreak attack success rate.



## **37. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**

cs.CL

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2402.16822v3) [paper-pdf](http://arxiv.org/pdf/2402.16822v3)

**Authors**: Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu

**Abstract**: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to adversarial attacks is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel black-box approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem and uses open-ended search to generate prompts that are both effective and diverse. Focusing on the safety domain, we use Rainbow Teaming to target various state-of-the-art LLMs, including the Llama 2 and Llama 3 models. Our approach reveals hundreds of effective adversarial prompts, with an attack success rate exceeding 90% across all tested models. Furthermore, we demonstrate that prompts generated by Rainbow Teaming are highly transferable and that fine-tuning models with synthetic data generated by our method significantly enhances their safety without sacrificing general performance or helpfulness. We additionally explore the versatility of Rainbow Teaming by applying it to question answering and cybersecurity, showcasing its potential to drive robust open-ended self-improvement in a wide range of applications.



## **38. Underestimated Privacy Risks for Minority Populations in Large Language Model Unlearning**

cs.LG

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08559v1) [paper-pdf](http://arxiv.org/pdf/2412.08559v1)

**Authors**: Rongzhe Wei, Mufei Li, Mohsen Ghassemi, Eleonora Kreačić, Yifan Li, Xiang Yue, Bo Li, Vamsi K. Potluru, Pan Li, Eli Chien

**Abstract**: Large Language Models are trained on extensive datasets that often contain sensitive, human-generated information, raising significant concerns about privacy breaches. While certified unlearning approaches offer strong privacy guarantees, they rely on restrictive model assumptions that are not applicable to LLMs. As a result, various unlearning heuristics have been proposed, with the associated privacy risks assessed only empirically. The standard evaluation pipelines typically randomly select data for removal from the training set, apply unlearning techniques, and use membership inference attacks to compare the unlearned models against models retrained without the to-be-unlearned data. However, since every data point is subject to the right to be forgotten, unlearning should be considered in the worst-case scenario from the privacy perspective. Prior work shows that data outliers may exhibit higher memorization effects. Intuitively, they are harder to be unlearn and thus the privacy risk of unlearning them is underestimated in the current evaluation. In this paper, we leverage minority data to identify such a critical flaw in previously widely adopted evaluations. We substantiate this claim through carefully designed experiments, including unlearning canaries related to minority groups, inspired by privacy auditing literature. Using personally identifiable information as a representative minority identifier, we demonstrate that minority groups experience at least 20% more privacy leakage in most cases across six unlearning approaches, three MIAs, three benchmark datasets, and two LLMs of different scales. Given that the right to be forgotten should be upheld for every individual, we advocate for a more rigorous evaluation of LLM unlearning methods. Our minority-aware evaluation framework represents an initial step toward ensuring more equitable assessments of LLM unlearning efficacy.



## **39. Model-Editing-Based Jailbreak against Safety-aligned Large Language Models**

cs.CR

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08201v1) [paper-pdf](http://arxiv.org/pdf/2412.08201v1)

**Authors**: Yuxi Li, Zhibo Zhang, Kailong Wang, Ling Shi, Haoyu Wang

**Abstract**: Large Language Models (LLMs) have transformed numerous fields by enabling advanced natural language interactions but remain susceptible to critical vulnerabilities, particularly jailbreak attacks. Current jailbreak techniques, while effective, often depend on input modifications, making them detectable and limiting their stealth and scalability. This paper presents Targeted Model Editing (TME), a novel white-box approach that bypasses safety filters by minimally altering internal model structures while preserving the model's intended functionalities. TME identifies and removes safety-critical transformations (SCTs) embedded in model matrices, enabling malicious queries to bypass restrictions without input modifications. By analyzing distinct activation patterns between safe and unsafe queries, TME isolates and approximates SCTs through an optimization process. Implemented in the D-LLM framework, our method achieves an average Attack Success Rate (ASR) of 84.86% on four mainstream open-source LLMs, maintaining high performance. Unlike existing methods, D-LLM eliminates the need for specific triggers or harmful response collections, offering a stealthier and more effective jailbreak strategy. This work reveals a covert and robust threat vector in LLM security and emphasizes the need for stronger safeguards in model safety alignment.



## **40. Doubly-Universal Adversarial Perturbations: Deceiving Vision-Language Models Across Both Images and Text with a Single Perturbation**

cs.CV

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08108v1) [paper-pdf](http://arxiv.org/pdf/2412.08108v1)

**Authors**: Hee-Seon Kim, Minbeom Kim, Changick Kim

**Abstract**: Large Vision-Language Models (VLMs) have demonstrated remarkable performance across multimodal tasks by integrating vision encoders with large language models (LLMs). However, these models remain vulnerable to adversarial attacks. Among such attacks, Universal Adversarial Perturbations (UAPs) are especially powerful, as a single optimized perturbation can mislead the model across various input images. In this work, we introduce a novel UAP specifically designed for VLMs: the Doubly-Universal Adversarial Perturbation (Doubly-UAP), capable of universally deceiving VLMs across both image and text inputs. To successfully disrupt the vision encoder's fundamental process, we analyze the core components of the attention mechanism. After identifying value vectors in the middle-to-late layers as the most vulnerable, we optimize Doubly-UAP in a label-free manner with a frozen model. Despite being developed as a black-box to the LLM, Doubly-UAP achieves high attack success rates on VLMs, consistently outperforming baseline methods across vision-language tasks. Extensive ablation studies and analyses further demonstrate the robustness of Doubly-UAP and provide insights into how it influences internal attention mechanisms.



## **41. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

cs.LG

11 pages, 5 figures

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08099v1) [paper-pdf](http://arxiv.org/pdf/2412.08099v1)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.



## **42. What You See Is Not Always What You Get: An Empirical Study of Code Comprehension by Large Language Models**

cs.SE

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08098v1) [paper-pdf](http://arxiv.org/pdf/2412.08098v1)

**Authors**: Bangshuo Zhu, Jiawen Wen, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering domain, covering numerous tasks such as code generation and comprehension. While the benefit of LLMs for coding task is well noted, it is perceived that LLMs are vulnerable to adversarial attacks. In this paper, we study the specific LLM vulnerability to imperceptible character attacks, a type of prompt-injection attack that uses special characters to befuddle an LLM whilst keeping the attack hidden to human eyes. We devise four categories of attacks and investigate their effects on the performance outcomes of tasks relating to code analysis and code comprehension. Two generations of ChatGPT are included to evaluate the impact of advancements made to contemporary models. Our experimental design consisted of comparing perturbed and unperturbed code snippets and evaluating two performance outcomes, which are model confidence using log probabilities of response, and correctness of response. We conclude that earlier version of ChatGPT exhibits a strong negative linear correlation between the amount of perturbation and the performance outcomes, while the recent ChatGPT presents a strong negative correlation between the presence of perturbation and performance outcomes, but no valid correlational relationship between perturbation budget and performance outcomes. We anticipate this work contributes to an in-depth understanding of leveraging LLMs for coding tasks. It is suggested future research should delve into how to create LLMs that can return a correct response even if the prompt exhibits perturbations.



## **43. Plentiful Jailbreaks with String Compositions**

cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2411.01084v3) [paper-pdf](http://arxiv.org/pdf/2411.01084v3)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or red-teamers, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary string compositions, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.



## **44. Summon a Demon and Bind it: A Grounded Theory of LLM Red Teaming**

cs.CL

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2311.06237v3) [paper-pdf](http://arxiv.org/pdf/2311.06237v3)

**Authors**: Nanna Inie, Jonathan Stray, Leon Derczynski

**Abstract**: Engaging in the deliberate generation of abnormal outputs from Large Language Models (LLMs) by attacking them is a novel human activity. This paper presents a thorough exposition of how and why people perform such attacks, defining LLM red-teaming based on extensive and diverse evidence. Using a formal qualitative methodology, we interviewed dozens of practitioners from a broad range of backgrounds, all contributors to this novel work of attempting to cause LLMs to fail. We focused on the research questions of defining LLM red teaming, uncovering the motivations and goals for performing the activity, and characterizing the strategies people use when attacking LLMs. Based on the data, LLM red teaming is defined as a limit-seeking, non-malicious, manual activity, which depends highly on a team-effort and an alchemist mindset. It is highly intrinsically motivated by curiosity, fun, and to some degrees by concerns for various harms of deploying LLMs. We identify a taxonomy of 12 strategies and 35 different techniques of attacking LLMs. These findings are presented as a comprehensive grounded theory of how and why people attack large language models: LLM red teaming.



## **45. FlexLLM: Exploring LLM Customization for Moving Target Defense on Black-Box LLMs Against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07672v1) [paper-pdf](http://arxiv.org/pdf/2412.07672v1)

**Authors**: Bocheng Chen, Hanqing Guo, Qiben Yan

**Abstract**: Defense in large language models (LLMs) is crucial to counter the numerous attackers exploiting these systems to generate harmful content through manipulated prompts, known as jailbreak attacks. Although many defense strategies have been proposed, they often require access to the model's internal structure or need additional training, which is impractical for service providers using LLM APIs, such as OpenAI APIs or Claude APIs. In this paper, we propose a moving target defense approach that alters decoding hyperparameters to enhance model robustness against various jailbreak attacks. Our approach does not require access to the model's internal structure and incurs no additional training costs. The proposed defense includes two key components: (1) optimizing the decoding strategy by identifying and adjusting decoding hyperparameters that influence token generation probabilities, and (2) transforming the decoding hyperparameters and model system prompts into dynamic targets, which are continuously altered during each runtime. By continuously modifying decoding strategies and prompts, the defense effectively mitigates the existing attacks. Our results demonstrate that our defense is the most effective against jailbreak attacks in three of the models tested when using LLMs as black-box APIs. Moreover, our defense offers lower inference costs and maintains comparable response quality, making it a potential layer of protection when used alongside other defense methods.



## **46. Look Before You Leap: Enhancing Attention and Vigilance Regarding Harmful Content with GuidelineLLM**

cs.CL

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.10423v1) [paper-pdf](http://arxiv.org/pdf/2412.10423v1)

**Authors**: Shaoqing Zhang, Zhuosheng Zhang, Kehai Chen, Rongxiang Weng, Muyun Yang, Tiejun Zhao, Min Zhang

**Abstract**: Despite being empowered with alignment mechanisms, large language models (LLMs) are increasingly vulnerable to emerging jailbreak attacks that can compromise their alignment mechanisms. This vulnerability poses significant risks to the real-world applications. Existing work faces challenges in both training efficiency and generalization capabilities (i.e., Reinforcement Learning from Human Feedback and Red-Teaming). Developing effective strategies to enable LLMs to resist continuously evolving jailbreak attempts represents a significant challenge. To address this challenge, we propose a novel defensive paradigm called GuidelineLLM, which assists LLMs in recognizing queries that may have harmful content. Before LLMs respond to a query, GuidelineLLM first identifies potential risks associated with the query, summarizes these risks into guideline suggestions, and then feeds these guidelines to the responding LLMs. Importantly, our approach eliminates the necessity for additional safety fine-tuning of the LLMs themselves; only the GuidelineLLM requires fine-tuning. This characteristic enhances the general applicability of GuidelineLLM across various LLMs. Experimental results demonstrate that GuidelineLLM can significantly reduce the attack success rate (ASR) against the LLMs (an average reduction of 34.17\% ASR) while maintaining the helpfulness of the LLMs in handling benign queries. Code is available at https://github.com/sqzhang-lazy/GuidelineLLM.



## **47. SQL Injection Jailbreak: a structural disaster of large language models**

cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2411.01565v3) [paper-pdf](http://arxiv.org/pdf/2411.01565v3)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality into various domains, generating substantial social and economic benefits. However, this swift advancement has also introduced new security vulnerabilities. Jailbreaking, a form of attack that induces LLMs to produce harmful content through carefully crafted prompts, presents a significant challenge to the safe and trustworthy development of LLMs. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak approaches and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. Our SIJ method achieves near 100\% attack success rates on five well-known open-source LLMs on the AdvBench, while incurring lower time costs compared to previous methods. More importantly, SIJ is the first method to exploit the external properties of LLMs for jailbreak attacks and exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at \href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.



## **48. MobileSafetyBench: Evaluating Safety of Autonomous Agents in Mobile Device Control**

cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2410.17520v2) [paper-pdf](http://arxiv.org/pdf/2410.17520v2)

**Authors**: Juyong Lee, Dongyoon Hahm, June Suk Choi, W. Bradley Knox, Kimin Lee

**Abstract**: Autonomous agents powered by large language models (LLMs) show promising potential in assistive tasks across various domains, including mobile device control. As these agents interact directly with personal information and device settings, ensuring their safe and reliable behavior is crucial to prevent undesirable outcomes. However, no benchmark exists for standardized evaluation of the safety of mobile device-control agents. In this work, we introduce MobileSafetyBench, a benchmark designed to evaluate the safety of device-control agents within a realistic mobile environment based on Android emulators. We develop a diverse set of tasks involving interactions with various mobile applications, including messaging and banking applications, challenging agents with managing risks encompassing misuse and negative side effects. These tasks include tests to evaluate the safety of agents in daily scenarios as well as their robustness against indirect prompt injection attacks. Our experiments demonstrate that baseline agents, based on state-of-the-art LLMs, often fail to effectively prevent harm while performing the tasks. To mitigate these safety concerns, we propose a prompting method that encourages agents to prioritize safety considerations. While this method shows promise in promoting safer behaviors, there is still considerable room for improvement to fully earn user trust. This highlights the urgent need for continued research to develop more robust safety mechanisms in mobile environments. We open-source our benchmark at: https://mobilesafetybench.github.io/.



## **49. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

cs.CL

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.12145v1) [paper-pdf](http://arxiv.org/pdf/2412.12145v1)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}



## **50. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07192v1) [paper-pdf](http://arxiv.org/pdf/2412.07192v1)

**Authors**: Zachary Coalson, Jeonghyun Woo, Shiyang Chen, Yu Sun, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.



