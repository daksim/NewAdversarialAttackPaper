# Latest Adversarial Attack Papers
**update at 2023-10-06 10:44:37**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03684v1) [paper-pdf](http://arxiv.org/pdf/2310.03684v1)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.



## **2. Misusing Tools in Large Language Models With Visual Adversarial Examples**

cs.CR

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.03185v1) [paper-pdf](http://arxiv.org/pdf/2310.03185v1)

**Authors**: Xiaohan Fu, Zihan Wang, Shuheng Li, Rajesh K. Gupta, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Models (LLMs) are being enhanced with the ability to use tools and to process multiple modalities. These new capabilities bring new benefits and also new security risks. In this work, we show that an attacker can use visual adversarial examples to cause attacker-desired tool usage. For example, the attacker could cause a victim LLM to delete calendar events, leak private conversations and book hotels. Different from prior work, our attacks can affect the confidentiality and integrity of user resources connected to the LLM while being stealthy and generalizable to multiple input prompts. We construct these attacks using gradient-based adversarial training and characterize performance along multiple dimensions. We find that our adversarial images can manipulate the LLM to invoke tools following real-world syntax almost always (~98%) while maintaining high similarity to clean images (~0.9 SSIM). Furthermore, using human scoring and automated metrics, we find that the attacks do not noticeably affect the conversation (and its semantics) between the user and the LLM.



## **3. LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples**

cs.CL

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.01469v2) [paper-pdf](http://arxiv.org/pdf/2310.01469v2)

**Authors**: Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Mu-Nan Ning, Li Yuan

**Abstract**: Large Language Models (LLMs), including GPT-3.5, LLaMA, and PaLM, seem to be knowledgeable and able to adapt to many tasks. However, we still can not completely trust their answer, since LLMs suffer from hallucination--fabricating non-existent facts to cheat users without perception. And the reasons for their existence and pervasiveness remain unclear. In this paper, we demonstrate that non-sense prompts composed of random tokens can also elicit the LLMs to respond with hallucinations. This phenomenon forces us to revisit that hallucination may be another view of adversarial examples, and it shares similar features with conventional adversarial examples as the basic feature of LLMs. Therefore, we formalize an automatic hallucination triggering method as the hallucination attack in an adversarial way. Finally, we explore basic feature of attacked adversarial prompts and propose a simple yet effective defense strategy. Our code is released on GitHub.



## **4. Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models**

cs.CL

Work in progress

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.02949v1) [paper-pdf](http://arxiv.org/pdf/2310.02949v1)

**Authors**: Xianjun Yang, Xiao Wang, Qi Zhang, Linda Petzold, William Yang Wang, Xun Zhao, Dahua Lin

**Abstract**: Warning: This paper contains examples of harmful language, and reader discretion is recommended. The increasing open release of powerful large language models (LLMs) has facilitated the development of downstream applications by reducing the essential cost of data annotation and computation. To ensure AI safety, extensive safety-alignment measures have been conducted to armor these models against malicious use (primarily hard prompt attack). However, beneath the seemingly resilient facade of the armor, there might lurk a shadow. By simply tuning on 100 malicious examples with 1 GPU hour, these safely aligned LLMs can be easily subverted to generate harmful content. Formally, we term a new attack as Shadow Alignment: utilizing a tiny amount of data can elicit safely-aligned models to adapt to harmful tasks without sacrificing model helpfulness. Remarkably, the subverted models retain their capability to respond appropriately to regular inquiries. Experiments across 8 models released by 5 different organizations (LLaMa-2, Falcon, InternLM, BaiChuan2, Vicuna) demonstrate the effectiveness of shadow alignment attack. Besides, the single-turn English-only attack successfully transfers to multi-turn dialogue and other languages. This study serves as a clarion call for a collective effort to overhaul and fortify the safety of open-source LLMs against malicious attackers.



## **5. DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text**

cs.CL

Updates

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2305.17359v2) [paper-pdf](http://arxiv.org/pdf/2305.17359v2)

**Authors**: Xianjun Yang, Wei Cheng, Yue Wu, Linda Petzold, William Yang Wang, Haifeng Chen

**Abstract**: Large language models (LLMs) have notably enhanced the fluency and diversity of machine-generated text. However, this progress also presents a significant challenge in detecting the origin of a given text, and current research on detection methods lags behind the rapid evolution of LLMs. Conventional training-based methods have limitations in flexibility, particularly when adapting to new domains, and they often lack explanatory power. To address this gap, we propose a novel training-free detection strategy called Divergent N-Gram Analysis (DNA-GPT). Given a text, we first truncate it in the middle and then use only the preceding portion as input to the LLMs to regenerate the new remaining parts. By analyzing the differences between the original and new remaining parts through N-gram analysis in black-box or probability divergence in white-box, we unveil significant discrepancies between the distribution of machine-generated text and the distribution of human-written text. We conducted extensive experiments on the most advanced LLMs from OpenAI, including text-davinci-003, GPT-3.5-turbo, and GPT-4, as well as open-source models such as GPT-NeoX-20B and LLaMa-13B. Results show that our zero-shot approach exhibits state-of-the-art performance in distinguishing between human and GPT-generated text on four English and one German dataset, outperforming OpenAI's own classifier, which is trained on millions of text. Additionally, our methods provide reasonable explanations and evidence to support our claim, which is a unique feature of explainable detection. Our method is also robust under the revised text attack and can additionally solve model sourcing. Codes are available at https://github.com/Xianjun-Yang/DNA-GPT.



## **6. Fewer is More: Trojan Attacks on Parameter-Efficient Fine-Tuning**

cs.CL

16 pages, 5 figures

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.00648v2) [paper-pdf](http://arxiv.org/pdf/2310.00648v2)

**Authors**: Lauren Hong, Ting Wang

**Abstract**: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of pre-trained language models (PLMs) to specific tasks. By tuning only a minimal set of (extra) parameters, PEFT achieves performance comparable to full fine-tuning. However, despite its prevalent use, the security implications of PEFT remain largely unexplored. In this paper, we conduct a pilot study revealing that PEFT exhibits unique vulnerability to trojan attacks. Specifically, we present PETA, a novel attack that accounts for downstream adaptation through bilevel optimization: the upper-level objective embeds the backdoor into a PLM while the lower-level objective simulates PEFT to retain the PLM's task-specific performance. With extensive evaluation across a variety of downstream tasks and trigger designs, we demonstrate PETA's effectiveness in terms of both attack success rate and unaffected clean accuracy, even after the victim user performs PEFT over the backdoored PLM using untainted data. Moreover, we empirically provide possible explanations for PETA's efficacy: the bilevel optimization inherently 'orthogonalizes' the backdoor and PEFT modules, thereby retaining the backdoor throughout PEFT. Based on this insight, we explore a simple defense that omits PEFT in selected layers of the backdoored PLM and unfreezes a subset of these layers' parameters, which is shown to effectively neutralize PETA.



## **7. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

cs.AI

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2309.10253v2) [paper-pdf](http://arxiv.org/pdf/2309.10253v2)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.



## **8. Low-Resource Languages Jailbreak GPT-4**

cs.CL

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02446v1) [paper-pdf](http://arxiv.org/pdf/2310.02446v1)

**Authors**: Zheng-Xin Yong, Cristina Menghini, Stephen H. Bach

**Abstract**: AI safety training and red-teaming of large language models (LLMs) are measures to mitigate the generation of unsafe content. Our work exposes the inherent cross-lingual vulnerability of these safety mechanisms, resulting from the linguistic inequality of safety training data, by successfully circumventing GPT-4's safeguard through translating unsafe English inputs into low-resource languages. On the AdvBenchmark, GPT-4 engages with the unsafe translated inputs and provides actionable items that can get the users towards their harmful goals 79% of the time, which is on par with or even surpassing state-of-the-art jailbreaking attacks. Other high-/mid-resource languages have significantly lower attack success rate, which suggests that the cross-lingual vulnerability mainly applies to low-resource languages. Previously, limited training on low-resource languages primarily affects speakers of those languages, causing technological disparities. However, our work highlights a crucial shift: this deficiency now poses a risk to all LLMs users. Publicly available translation APIs enable anyone to exploit LLMs' safety vulnerabilities. Therefore, our work calls for a more holistic red-teaming efforts to develop robust multilingual safeguards with wide language coverage.



## **9. Jailbreaker in Jail: Moving Target Defense for Large Language Models**

cs.CR

MTD Workshop in CCS'23

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02417v1) [paper-pdf](http://arxiv.org/pdf/2310.02417v1)

**Authors**: Bocheng Chen, Advait Paliwal, Qiben Yan

**Abstract**: Large language models (LLMs), known for their capability in understanding and following instructions, are vulnerable to adversarial attacks. Researchers have found that current commercial LLMs either fail to be "harmless" by presenting unethical answers, or fail to be "helpful" by refusing to offer meaningful answers when faced with adversarial queries. To strike a balance between being helpful and harmless, we design a moving target defense (MTD) enhanced LLM system. The system aims to deliver non-toxic answers that align with outputs from multiple model candidates, making them more robust against adversarial attacks. We design a query and output analysis model to filter out unsafe or non-responsive answers. %to achieve the two objectives of randomly selecting outputs from different LLMs. We evaluate over 8 most recent chatbot models with state-of-the-art adversarial queries. Our MTD-enhanced LLM system reduces the attack success rate from 37.5\% to 0\%. Meanwhile, it decreases the response refusal rate from 50\% to 0\%.



## **10. Gotcha! This Model Uses My Code! Evaluating Membership Leakage Risks in Code Models**

cs.SE

13 pages

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2310.01166v1) [paper-pdf](http://arxiv.org/pdf/2310.01166v1)

**Authors**: Zhou Yang, Zhipeng Zhao, Chenyu Wang, Jieke Shi, Dongsum Kim, Donggyun Han, David Lo

**Abstract**: Given large-scale source code datasets available in open-source projects and advanced large language models, recent code models have been proposed to address a series of critical software engineering tasks, such as program repair and code completion. The training data of the code models come from various sources, not only the publicly available source code, e.g., open-source projects on GitHub but also the private data such as the confidential source code from companies, which may contain sensitive information (for example, SSH keys and personal information). As a result, the use of these code models may raise new privacy concerns.   In this paper, we focus on a critical yet not well-explored question on using code models: what is the risk of membership information leakage in code models? Membership information leakage refers to the risk that an attacker can infer whether a given data point is included in (i.e., a member of) the training data. To answer this question, we propose Gotcha, a novel membership inference attack method specifically for code models. We investigate the membership leakage risk of code models. Our results reveal a worrying fact that the risk of membership leakage is high: although the previous attack methods are close to random guessing, Gotcha can predict the data membership with a high true positive rate of 0.95 and a low false positive rate of 0.10. We also show that the attacker's knowledge of the victim model (e.g., the model architecture and the pre-training data) impacts the success rate of attacks. Further analysis demonstrates that changing the decoding strategy can mitigate the risk of membership leakage. This study calls for more attention to understanding the privacy of code models and developing more effective countermeasures against such attacks.



## **11. LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud**

cs.CL

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2309.17157v2) [paper-pdf](http://arxiv.org/pdf/2309.17157v2)

**Authors**: Mengke Zhang, Tianxing He, Tianle Wang, Lu Mi, Fatemehsadat Mireshghallah, Binyi Chen, Hao Wang, Yulia Tsvetkov

**Abstract**: In the current user-server interaction paradigm of prompted generation with large language models (LLM) on cloud, the server fully controls the generation process, which leaves zero options for users who want to keep the generated text to themselves. We propose LatticeGen, a cooperative framework in which the server still handles most of the computation while the user controls the sampling operation. The key idea is that the true generated sequence is mixed with noise tokens by the user and hidden in a noised lattice. Considering potential attacks from a hypothetically malicious server and how the user can defend against it, we propose the repeated beam-search attack and the mixing noise scheme. In our experiments we apply LatticeGen to protect both prompt and generation. It is shown that while the noised lattice degrades generation quality, LatticeGen successfully protects the true generation to a remarkable degree under strong attacks (more than 50% of the semantic remains hidden as measured by BERTScore).



## **12. Streamlining Attack Tree Generation: A Fragment-Based Approach**

cs.CR

To appear at the 57th Hawaii International Conference on Social  Systems (HICSS-57), Honolulu, Hawaii. 2024

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00654v1) [paper-pdf](http://arxiv.org/pdf/2310.00654v1)

**Authors**: Irdin Pekaric, Markus Frick, Jubril Gbolahan Adigun, Raffaela Groner, Thomas Witte, Alexander Raschke, Michael Felderer, Matthias Tichy

**Abstract**: Attack graphs are a tool for analyzing security vulnerabilities that capture different and prospective attacks on a system. As a threat modeling tool, it shows possible paths that an attacker can exploit to achieve a particular goal. However, due to the large number of vulnerabilities that are published on a daily basis, they have the potential to rapidly expand in size. Consequently, this necessitates a significant amount of resources to generate attack graphs. In addition, generating composited attack models for complex systems such as self-adaptive or AI is very difficult due to their nature to continuously change. In this paper, we present a novel fragment-based attack graph generation approach that utilizes information from publicly available information security databases. Furthermore, we also propose a domain-specific language for attack modeling, which we employ in the proposed attack graph generation approach. Finally, we present a demonstrator example showcasing the attack generator's capability to replicate a verified attack chain, as previously confirmed by security experts.



## **13. Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection**

cs.CL

The data and code can be found at  https://github.com/Leezekun/Adv-Instruct-Eval

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2308.10819v2) [paper-pdf](http://arxiv.org/pdf/2308.10819v2)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency in following instructions, making them valuable in customer-facing applications. However, their impressive capabilities also raise concerns about the amplification of risks posed by adversarial instructions, which can be injected into the model input by third-party attackers to manipulate LLMs' original instructions and prompt unintended actions and content. Therefore, it is crucial to understand LLMs' ability to accurately discern which instructions to follow to ensure their safe deployment in real-world scenarios. In this paper, we propose a pioneering benchmark for automatically evaluating the robustness of instruction-following LLMs against adversarial instructions injected in the prompt. The objective of this benchmark is to quantify the extent to which LLMs are influenced by injected adversarial instructions and assess their ability to differentiate between these injected adversarial instructions and original user instructions. Through experiments conducted with state-of-the-art instruction-following LLMs, we uncover significant limitations in their robustness against adversarial instruction injection attacks. Furthermore, our findings indicate that prevalent instruction-tuned models are prone to being ``overfitted'' to follow any instruction phrase in the prompt without truly understanding which instructions should be followed. This highlights the need to address the challenge of training models to comprehend prompts instead of merely following instruction phrases and completing the text. The data and code can be found at \url{https://github.com/Leezekun/Adv-Instruct-Eval}.



## **14. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

cs.CL

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2310.00322v1) [paper-pdf](http://arxiv.org/pdf/2310.00322v1)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.



## **15. FLIP: Cross-domain Face Anti-spoofing with Language Guidance**

cs.CV

Accepted to ICCV-2023. Project Page:  https://koushiksrivats.github.io/FLIP/

**SubmitDate**: 2023-09-28    [abs](http://arxiv.org/abs/2309.16649v1) [paper-pdf](http://arxiv.org/pdf/2309.16649v1)

**Authors**: Koushik Srivatsan, Muzammal Naseer, Karthik Nandakumar

**Abstract**: Face anti-spoofing (FAS) or presentation attack detection is an essential component of face recognition systems deployed in security-critical applications. Existing FAS methods have poor generalizability to unseen spoof types, camera sensors, and environmental conditions. Recently, vision transformer (ViT) models have been shown to be effective for the FAS task due to their ability to capture long-range dependencies among image patches. However, adaptive modules or auxiliary loss functions are often required to adapt pre-trained ViT weights learned on large-scale datasets such as ImageNet. In this work, we first show that initializing ViTs with multimodal (e.g., CLIP) pre-trained weights improves generalizability for the FAS task, which is in line with the zero-shot transfer capabilities of vision-language pre-trained (VLP) models. We then propose a novel approach for robust cross-domain FAS by grounding visual representations with the help of natural language. Specifically, we show that aligning the image representation with an ensemble of class descriptions (based on natural language semantics) improves FAS generalizability in low-data regimes. Finally, we propose a multimodal contrastive learning strategy to boost feature generalization further and bridge the gap between source and target domains. Extensive experiments on three standard protocols demonstrate that our method significantly outperforms the state-of-the-art methods, achieving better zero-shot transfer performance than five-shot transfer of adaptive ViTs. Code: https://github.com/koushiksrivats/FLIP



## **16. VDC: Versatile Data Cleanser for Detecting Dirty Samples via Visual-Linguistic Inconsistency**

cs.CV

22 pages,5 figures,17 tables

**SubmitDate**: 2023-09-28    [abs](http://arxiv.org/abs/2309.16211v1) [paper-pdf](http://arxiv.org/pdf/2309.16211v1)

**Authors**: Zihao Zhu, Mingda Zhang, Shaokui Wei, Bingzhe Wu, Baoyuan Wu

**Abstract**: The role of data in building AI systems has recently been emphasized by the emerging concept of data-centric AI. Unfortunately, in the real-world, datasets may contain dirty samples, such as poisoned samples from backdoor attack, noisy labels in crowdsourcing, and even hybrids of them. The presence of such dirty samples makes the DNNs vunerable and unreliable.Hence, it is critical to detect dirty samples to improve the quality and realiability of dataset. Existing detectors only focus on detecting poisoned samples or noisy labels, that are often prone to weak generalization when dealing with dirty samples from other domains.In this paper, we find a commonality of various dirty samples is visual-linguistic inconsistency between images and associated labels. To capture the semantic inconsistency between modalities, we propose versatile data cleanser (VDC) leveraging the surpassing capabilities of multimodal large language models (MLLM) in cross-modal alignment and reasoning.It consists of three consecutive modules: the visual question generation module to generate insightful questions about the image; the visual question answering module to acquire the semantics of the visual content by answering the questions with MLLM; followed by the visual answer evaluation module to evaluate the inconsistency.Extensive experiments demonstrate its superior performance and generalization to various categories and types of dirty samples.



## **17. Advancing Beyond Identification: Multi-bit Watermark for Large Language Models**

cs.CL

Under review. 9 pages and appendix

**SubmitDate**: 2023-09-27    [abs](http://arxiv.org/abs/2308.00221v2) [paper-pdf](http://arxiv.org/pdf/2308.00221v2)

**Authors**: KiYoon Yoo, Wonhyuk Ahn, Nojun Kwak

**Abstract**: We propose a method to tackle misuses of large language models beyond the identification of machine-generated text. While existing methods focus on detection, some malicious misuses demand tracing the adversary user for counteracting them. To address this, we propose Multi-bit Watermark via Position Allocation, embedding traceable multi-bit information during language model generation. Leveraging the benefits of zero-bit watermarking, our method enables robust extraction of the watermark without any model access, embedding and extraction of long messages ($\geq$ 32-bit) without finetuning, and maintaining text quality, while allowing zero-bit detection all at the same time. Moreover, our watermark is relatively robust under strong attacks like interleaving human texts and paraphrasing.



## **18. Large Language Model Alignment: A Survey**

cs.CL

76 pages

**SubmitDate**: 2023-09-26    [abs](http://arxiv.org/abs/2309.15025v1) [paper-pdf](http://arxiv.org/pdf/2309.15025v1)

**Authors**: Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, Deyi Xiong

**Abstract**: Recent years have witnessed remarkable progress made in large language models (LLMs). Such advancements, while garnering significant attention, have concurrently elicited various concerns. The potential of these models is undeniably vast; however, they may yield texts that are imprecise, misleading, or even detrimental. Consequently, it becomes paramount to employ alignment techniques to ensure these models to exhibit behaviors consistent with human values.   This survey endeavors to furnish an extensive exploration of alignment methodologies designed for LLMs, in conjunction with the extant capability research in this domain. Adopting the lens of AI alignment, we categorize the prevailing methods and emergent proposals for the alignment of LLMs into outer and inner alignment. We also probe into salient issues including the models' interpretability, and potential vulnerabilities to adversarial attacks. To assess LLM alignment, we present a wide variety of benchmarks and evaluation methodologies. After discussing the state of alignment research for LLMs, we finally cast a vision toward the future, contemplating the promising avenues of research that lie ahead.   Our aspiration for this survey extends beyond merely spurring research interests in this realm. We also envision bridging the gap between the AI alignment research community and the researchers engrossed in the capability exploration of LLMs for both capable and safe LLMs.



## **19. SurrogatePrompt: Bypassing the Safety Filter of Text-To-Image Models via Substitution**

cs.CV

14 pages, 11 figures

**SubmitDate**: 2023-09-25    [abs](http://arxiv.org/abs/2309.14122v1) [paper-pdf](http://arxiv.org/pdf/2309.14122v1)

**Authors**: Zhongjie Ba, Jieming Zhong, Jiachen Lei, Peng Cheng, Qinglong Wang, Zhan Qin, Zhibo Wang, Kui Ren

**Abstract**: Advanced text-to-image models such as DALL-E 2 and Midjourney possess the capacity to generate highly realistic images, raising significant concerns regarding the potential proliferation of unsafe content. This includes adult, violent, or deceptive imagery of political figures. Despite claims of rigorous safety mechanisms implemented in these models to restrict the generation of not-safe-for-work (NSFW) content, we successfully devise and exhibit the first prompt attacks on Midjourney, resulting in the production of abundant photorealistic NSFW images. We reveal the fundamental principles of such prompt attacks and suggest strategically substituting high-risk sections within a suspect prompt to evade closed-source safety measures. Our novel framework, SurrogatePrompt, systematically generates attack prompts, utilizing large language models, image-to-text, and image-to-image modules to automate attack prompt creation at scale. Evaluation results disclose an 88% success rate in bypassing Midjourney's proprietary safety filter with our attack prompts, leading to the generation of counterfeit images depicting political figures in violent scenarios. Both subjective and objective assessments validate that the images generated from our attack prompts present considerable safety hazards.



## **20. FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and LLMs**

cs.CR

**SubmitDate**: 2023-09-25    [abs](http://arxiv.org/abs/2306.04959v2) [paper-pdf](http://arxiv.org/pdf/2306.04959v2)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedMLSecurity, a benchmark designed to simulate adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). As an integral module of the open-sourced library FedML that facilitates FL algorithm development and performance comparison, FedMLSecurity enhances FedML's capabilities to evaluate security issues and potential remedies in FL. FedMLSecurity comprises two major components: FedMLAttacker that simulates attacks injected during FL training, and FedMLDefender that simulates defensive mechanisms to mitigate the impacts of the attacks. FedMLSecurity is open-sourced and can be customized to a wide range of machine learning models (e.g., Logistic Regression, ResNet, GAN, etc.) and federated optimizers (e.g., FedAVG, FedOPT, FedNOVA, etc.). FedMLSecurity can also be applied to Large Language Models (LLMs) easily, demonstrating its adaptability and applicability in various scenarios.



## **21. Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks**

cs.LG

Accepted by NeurIPS'23

**SubmitDate**: 2023-09-23    [abs](http://arxiv.org/abs/2309.13256v1) [paper-pdf](http://arxiv.org/pdf/2309.13256v1)

**Authors**: Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Jinghui Chen, Fenglong Ma, Ting Wang

**Abstract**: Pre-trained language models (PLMs) have demonstrated remarkable performance as few-shot learners. However, their security risks under such settings are largely unexplored. In this work, we conduct a pilot study showing that PLMs as few-shot learners are highly vulnerable to backdoor attacks while existing defenses are inadequate due to the unique challenges of few-shot scenarios. To address such challenges, we advocate MDP, a novel lightweight, pluggable, and effective defense for PLMs as few-shot learners. Specifically, MDP leverages the gap between the masking-sensitivity of poisoned and clean samples: with reference to the limited few-shot data as distributional anchors, it compares the representations of given samples under varying masking and identifies poisoned samples as ones with significant variations. We show analytically that MDP creates an interesting dilemma for the attacker to choose between attack effectiveness and detection evasiveness. The empirical evaluation using benchmark datasets and representative attacks validates the efficacy of MDP.



## **22. Knowledge Sanitization of Large Language Models**

cs.CL

**SubmitDate**: 2023-09-21    [abs](http://arxiv.org/abs/2309.11852v1) [paper-pdf](http://arxiv.org/pdf/2309.11852v1)

**Authors**: Yoichi Ishibashi, Hidetoshi Shimodaira

**Abstract**: We explore a knowledge sanitization approach to mitigate the privacy concerns associated with large language models (LLMs). LLMs trained on a large corpus of Web data can memorize and potentially reveal sensitive or confidential information, raising critical security concerns. Our technique fine-tunes these models, prompting them to generate harmless responses such as ``I don't know'' when queried about specific information. Experimental results in a closed-book question-answering task show that our straightforward method not only minimizes particular knowledge leakage but also preserves the overall performance of LLM. These two advantages strengthen the defense against extraction attacks and reduces the emission of harmful content such as hallucinations.



## **23. A Chinese Prompt Attack Dataset for LLMs with Evil Content**

cs.CL

**SubmitDate**: 2023-09-21    [abs](http://arxiv.org/abs/2309.11830v1) [paper-pdf](http://arxiv.org/pdf/2309.11830v1)

**Authors**: Chengyuan Liu, Fubang Zhao, Lizhi Qing, Yangyang Kang, Changlong Sun, Kun Kuang, Fei Wu

**Abstract**: Large Language Models (LLMs) present significant priority in text understanding and generation. However, LLMs suffer from the risk of generating harmful contents especially while being employed to applications. There are several black-box attack methods, such as Prompt Attack, which can change the behaviour of LLMs and induce LLMs to generate unexpected answers with harmful contents. Researchers are interested in Prompt Attack and Defense with LLMs, while there is no publicly available dataset to evaluate the abilities of defending prompt attack. In this paper, we introduce a Chinese Prompt Attack Dataset for LLMs, called CPAD. Our prompts aim to induce LLMs to generate unexpected outputs with several carefully designed prompt attack approaches and widely concerned attacking contents. Different from previous datasets involving safety estimation, We construct the prompts considering three dimensions: contents, attacking methods and goals, thus the responses can be easily evaluated and analysed. We run several well-known Chinese LLMs on our dataset, and the results show that our prompts are significantly harmful to LLMs, with around 70% attack success rate. We will release CPAD to encourage further studies on prompt attack and defense.



## **24. How Robust is Google's Bard to Adversarial Image Attacks?**

cs.CV

Technical report

**SubmitDate**: 2023-09-21    [abs](http://arxiv.org/abs/2309.11751v1) [paper-pdf](http://arxiv.org/pdf/2309.11751v1)

**Authors**: Yinpeng Dong, Huanran Chen, Jiawei Chen, Zhengwei Fang, Xiao Yang, Yichi Zhang, Yu Tian, Hang Su, Jun Zhu

**Abstract**: Multimodal Large Language Models (MLLMs) that integrate text and other modalities (especially vision) have achieved unprecedented performance in various multimodal tasks. However, due to the unsolved adversarial robustness problem of vision models, MLLMs can have more severe safety and security risks by introducing the vision inputs. In this work, we study the adversarial robustness of Google's Bard, a competitive chatbot to ChatGPT that released its multimodal capability recently, to better understand the vulnerabilities of commercial MLLMs. By attacking white-box surrogate vision encoders or MLLMs, the generated adversarial examples can mislead Bard to output wrong image descriptions with a 22% success rate based solely on the transferability. We show that the adversarial examples can also attack other MLLMs, e.g., a 26% attack success rate against Bing Chat and a 86% attack success rate against ERNIE bot. Moreover, we identify two defense mechanisms of Bard, including face detection and toxicity detection of images. We design corresponding attacks to evade these defenses, demonstrating that the current defenses of Bard are also vulnerable. We hope this work can deepen our understanding on the robustness of MLLMs and facilitate future research on defenses. Our code is available at https://github.com/thu-ml/Attack-Bard.



## **25. Model Leeching: An Extraction Attack Targeting LLMs**

cs.LG

**SubmitDate**: 2023-09-19    [abs](http://arxiv.org/abs/2309.10544v1) [paper-pdf](http://arxiv.org/pdf/2309.10544v1)

**Authors**: Lewis Birch, William Hackett, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Model Leeching is a novel extraction attack targeting Large Language Models (LLMs), capable of distilling task-specific knowledge from a target LLM into a reduced parameter model. We demonstrate the effectiveness of our attack by extracting task capability from ChatGPT-3.5-Turbo, achieving 73% Exact Match (EM) similarity, and SQuAD EM and F1 accuracy scores of 75% and 87%, respectively for only $50 in API cost. We further demonstrate the feasibility of adversarial attack transferability from an extracted model extracted via Model Leeching to perform ML attack staging against a target LLM, resulting in an 11% increase to attack success rate when applied to ChatGPT-3.5-Turbo.



## **26. Language Guided Adversarial Purification**

cs.LG

**SubmitDate**: 2023-09-19    [abs](http://arxiv.org/abs/2309.10348v1) [paper-pdf](http://arxiv.org/pdf/2309.10348v1)

**Authors**: Himanshu Singh, A V Subramanyam

**Abstract**: Adversarial purification using generative models demonstrates strong adversarial defense performance. These methods are classifier and attack-agnostic, making them versatile but often computationally intensive. Recent strides in diffusion and score networks have improved image generation and, by extension, adversarial purification. Another highly efficient class of adversarial defense methods known as adversarial training requires specific knowledge of attack vectors, forcing them to be trained extensively on adversarial examples. To overcome these limitations, we introduce a new framework, namely Language Guided Adversarial Purification (LGAP), utilizing pre-trained diffusion models and caption generators to defend against adversarial attacks. Given an input image, our method first generates a caption, which is then used to guide the adversarial purification process through a diffusion network. Our approach has been evaluated against strong adversarial attacks, proving its effectiveness in enhancing adversarial robustness. Our results indicate that LGAP outperforms most existing adversarial defense techniques without requiring specialized network training. This underscores the generalizability of models trained on large datasets, highlighting a promising direction for further research.



## **27. LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins**

cs.CR

**SubmitDate**: 2023-09-19    [abs](http://arxiv.org/abs/2309.10254v1) [paper-pdf](http://arxiv.org/pdf/2309.10254v1)

**Authors**: Umar Iqbal, Tadayoshi Kohno, Franziska Roesner

**Abstract**: Large language model (LLM) platforms, such as ChatGPT, have recently begun offering a plugin ecosystem to interface with third-party services on the internet. While these plugins extend the capabilities of LLM platforms, they are developed by arbitrary third parties and thus cannot be implicitly trusted. Plugins also interface with LLM platforms and users using natural language, which can have imprecise interpretations. In this paper, we propose a framework that lays a foundation for LLM platform designers to analyze and improve the security, privacy, and safety of current and future plugin-integrated LLM platforms. Our framework is a formulation of an attack taxonomy that is developed by iteratively exploring how LLM platform stakeholders could leverage their capabilities and responsibilities to mount attacks against each other. As part of our iterative process, we apply our framework in the context of OpenAI's plugin ecosystem. We uncover plugins that concretely demonstrate the potential for the types of issues that we outline in our attack taxonomy. We conclude by discussing novel challenges and by providing recommendations to improve the security, privacy, and safety of present and future LLM-based computing platforms.



## **28. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

cs.CL

16 Pages, 5 Figures, 3 Tables

**SubmitDate**: 2023-09-18    [abs](http://arxiv.org/abs/2309.14348v1) [paper-pdf](http://arxiv.org/pdf/2309.14348v1)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100\% to around 10\% or less.



## **29. Your Room is not Private: Gradient Inversion Attack on Reinforcement Learning**

cs.RO

7 pages, 4 figures, 2 tables

**SubmitDate**: 2023-09-17    [abs](http://arxiv.org/abs/2306.09273v2) [paper-pdf](http://arxiv.org/pdf/2306.09273v2)

**Authors**: Miao Li, Wenhao Ding, Ding Zhao

**Abstract**: The prominence of embodied Artificial Intelligence (AI), which empowers robots to navigate, perceive, and engage within virtual environments, has attracted significant attention, owing to the remarkable advancements in computer vision and large language models. Privacy emerges as a pivotal concern within the realm of embodied AI, as the robot accesses substantial personal information. However, the issue of privacy leakage in embodied AI tasks, particularly in relation to reinforcement learning algorithms, has not received adequate consideration in research. This paper aims to address this gap by proposing an attack on the value-based algorithm and the gradient-based algorithm, utilizing gradient inversion to reconstruct states, actions, and supervision signals. The choice of using gradients for the attack is motivated by the fact that commonly employed federated learning techniques solely utilize gradients computed based on private user data to optimize models, without storing or transmitting the data to public servers. Nevertheless, these gradients contain sufficient information to potentially expose private data. To validate our approach, we conduct experiments on the AI2THOR simulator and evaluate our algorithm on active perception, a prevalent task in embodied AI. The experimental results demonstrate the effectiveness of our method in successfully reconstructing all information from the data across 120 room layouts.



## **30. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

cs.CL

**SubmitDate**: 2023-09-17    [abs](http://arxiv.org/abs/2309.01446v2) [paper-pdf](http://arxiv.org/pdf/2309.01446v2)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.



## **31. Context-aware Adversarial Attack on Named Entity Recognition**

cs.CL

**SubmitDate**: 2023-09-16    [abs](http://arxiv.org/abs/2309.08999v1) [paper-pdf](http://arxiv.org/pdf/2309.08999v1)

**Authors**: Shuguang Chen, Leonardo Neves, Thamar Solorio

**Abstract**: In recent years, large pre-trained language models (PLMs) have achieved remarkable performance on many natural language processing benchmarks. Despite their success, prior studies have shown that PLMs are vulnerable to attacks from adversarial examples. In this work, we focus on the named entity recognition task and study context-aware adversarial attack methods to examine the model's robustness. Specifically, we propose perturbing the most informative words for recognizing entities to create adversarial examples and investigate different candidate replacement methods to generate natural and plausible adversarial examples. Experiments and analyses show that our methods are more effective in deceiving the model into making wrong predictions than strong baselines.



## **32. ICLEF: In-Context Learning with Expert Feedback for Explainable Style Transfer**

cs.CL

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2309.08583v1) [paper-pdf](http://arxiv.org/pdf/2309.08583v1)

**Authors**: Arkadiy Saakyan, Smaranda Muresan

**Abstract**: While state-of-the-art language models excel at the style transfer task, current work does not address explainability of style transfer systems. Explanations could be generated using large language models such as GPT-3.5 and GPT-4, but the use of such complex systems is inefficient when smaller, widely distributed, and transparent alternatives are available. We propose a framework to augment and improve a formality style transfer dataset with explanations via model distillation from ChatGPT. To further refine the generated explanations, we propose a novel way to incorporate scarce expert human feedback using in-context learning (ICLEF: In-Context Learning from Expert Feedback) by prompting ChatGPT to act as a critic to its own outputs. We use the resulting dataset of 9,960 explainable formality style transfer instances (e-GYAFC) to show that current openly distributed instruction-tuned models (and, in some settings, ChatGPT) perform poorly on the task, and that fine-tuning on our high-quality dataset leads to significant improvements as shown by automatic evaluation. In human evaluation, we show that models much smaller than ChatGPT fine-tuned on our data align better with expert preferences. Finally, we discuss two potential applications of models fine-tuned on the explainable style transfer task: interpretable authorship verification and interpretable adversarial attacks on AI-generated text detectors.



## **33. Adversarial Attacks on Tables with Entity Swap**

cs.CL

Accepted at TaDA workshop at VLDB 2023

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2309.08650v1) [paper-pdf](http://arxiv.org/pdf/2309.08650v1)

**Authors**: Aneta Koleva, Martin Ringsquandl, Volker Tresp

**Abstract**: The capabilities of large language models (LLMs) have been successfully applied in the context of table representation learning. The recently proposed tabular language models have reported state-of-the-art results across various tasks for table interpretation. However, a closer look into the datasets commonly used for evaluation reveals an entity leakage from the train set into the test set. Motivated by this observation, we explore adversarial attacks that represent a more realistic inference setup. Adversarial attacks on text have been shown to greatly affect the performance of LLMs, but currently, there are no attacks targeting tabular language models. In this paper, we propose an evasive entity-swap attack for the column type annotation (CTA) task. Our CTA attack is the first black-box attack on tables, where we employ a similarity-based sampling strategy to generate adversarial examples. The experimental results show that the proposed attack generates up to a 70% drop in performance.



## **34. RAIN: Your Language Models Can Align Themselves without Finetuning**

cs.CL

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.07124v1) [paper-pdf](http://arxiv.org/pdf/2309.07124v1)

**Authors**: Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, Hongyang Zhang

**Abstract**: Large language models (LLMs) often demonstrate inconsistencies with human preferences. Previous research gathered human preference data and then aligned the pre-trained models using reinforcement learning or instruction tuning, the so-called finetuning step. In contrast, aligning frozen LLMs without any extra data is more appealing. This work explores the potential of the latter setting. We discover that by integrating self-evaluation and rewind mechanisms, unaligned LLMs can directly produce responses consistent with human preferences via self-boosting. We introduce a novel inference method, Rewindable Auto-regressive INference (RAIN), that allows pre-trained LLMs to evaluate their own generation and use the evaluation results to guide backward rewind and forward generation for AI safety. Notably, RAIN operates without the need of extra data for model alignment and abstains from any training, gradient computation, or parameter updates; during the self-evaluation phase, the model receives guidance on which human preference to align with through a fixed-template prompt, eliminating the need to modify the initial prompt. Experimental results evaluated by GPT-4 and humans demonstrate the effectiveness of RAIN: on the HH dataset, RAIN improves the harmlessness rate of LLaMA 30B over vanilla inference from 82% to 97%, while maintaining the helpfulness rate. Under the leading adversarial attack llm-attacks on Vicuna 33B, RAIN establishes a new defense baseline by reducing the attack success rate from 94% to 19%.



## **35. Games and Argumentation: Time for a Family Reunion!**

cs.LO

Fourth Workshop on Explainable Logic-Based Knowledge Representation  (XLoKR), Sept 2, 2023. Rhodes, Greece

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2309.06620v1) [paper-pdf](http://arxiv.org/pdf/2309.06620v1)

**Authors**: Bertram Ludäscher, Yilin Xia

**Abstract**: The rule "defeated(X) $\leftarrow$ attacks(Y,X), $\neg$ defeated(Y)" states that an argument is defeated if it is attacked by an argument that is not defeated. The rule "win(X) $\leftarrow$ move(X,Y), $\neg$ win(Y)" states that in a game a position is won if there is a move to a position that is not won. Both logic rules can be seen as close relatives (even identical twins) and both rules have been at the center of attention at various times in different communities: The first rule lies at the core of argumentation frameworks and has spawned a large family of models and semantics of abstract argumentation. The second rule has played a key role in the quest to find the "right" semantics for logic programs with recursion through negation, and has given rise to the stable and well-founded semantics. Both semantics have been widely studied by the logic programming and nonmonotonic reasoning community. The second rule has also received much attention by the database and finite model theory community, e.g., when studying the expressive power of query languages and fixpoint logics. Although close connections between argumentation frameworks, logic programming, and dialogue games have been known for a long time, the overlap and cross-fertilization between the communities appears to be smaller than one might expect. To this end, we recall some of the key results from database theory in which the win-move query has played a central role, e.g., on normal forms and expressive power of query languages. We introduce some notions that naturally emerge from games and that may provide new perspectives and research opportunities for argumentation frameworks. We discuss how solved query evaluation games reveal how- and why-not provenance of query answers. These techniques can be used to explain how results were derived via the given query, game, or argumentation framework.



## **36. FuzzLLM: A Novel and Universal Fuzzing Framework for Proactively Discovering Jailbreak Vulnerabilities in Large Language Models**

cs.CR

In submission, a preprint version

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2309.05274v1) [paper-pdf](http://arxiv.org/pdf/2309.05274v1)

**Authors**: Dongyu Yao, Jianshu Zhang, Ian G. Harris, Marcel Carlsson

**Abstract**: Jailbreak vulnerabilities in Large Language Models (LLMs), which exploit meticulously crafted prompts to elicit content that violates service guidelines, have captured the attention of research communities. While model owners can defend against individual jailbreak prompts through safety training strategies, this relatively passive approach struggles to handle the broader category of similar jailbreaks. To tackle this issue, we introduce FuzzLLM, an automated fuzzing framework designed to proactively test and discover jailbreak vulnerabilities in LLMs. We utilize templates to capture the structural integrity of a prompt and isolate key features of a jailbreak class as constraints. By integrating different base classes into powerful combo attacks and varying the elements of constraints and prohibited questions, FuzzLLM enables efficient testing with reduced manual effort. Extensive experiments demonstrate FuzzLLM's effectiveness and comprehensiveness in vulnerability discovery across various LLMs.



## **37. RatGPT: Turning online LLMs into Proxies for Malware Attacks**

cs.CR

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2308.09183v2) [paper-pdf](http://arxiv.org/pdf/2308.09183v2)

**Authors**: Mika Beckerich, Laura Plein, Sergio Coronado

**Abstract**: The evolution of Generative AI and the capabilities of the newly released Large Language Models (LLMs) open new opportunities in software engineering. However, they also lead to new challenges in cybersecurity. Recently, researchers have shown the possibilities of using LLMs such as ChatGPT to generate malicious content that can directly be exploited or guide inexperienced hackers to weaponize tools and code. These studies covered scenarios that still require the attacker to be in the middle of the loop. In this study, we leverage openly available plugins and use an LLM as proxy between the attacker and the victim. We deliver a proof-of-concept where ChatGPT is used for the dissemination of malicious software while evading detection, alongside establishing the communication to a command and control (C2) server to receive commands to interact with a victim's system. Finally, we present the general approach as well as essential elements in order to stay undetected and make the attack a success. This proof-of-concept highlights significant cybersecurity issues with openly available plugins and LLMs, which require the development of security guidelines, controls, and mitigation strategies.



## **38. Demystifying RCE Vulnerabilities in LLM-Integrated Apps**

cs.CR

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.02926v1) [paper-pdf](http://arxiv.org/pdf/2309.02926v1)

**Authors**: Tong Liu, Zizhuang Deng, Guozhu Meng, Yuekang Li, Kai Chen

**Abstract**: In recent years, Large Language Models (LLMs) have demonstrated remarkable potential across various downstream tasks. LLM-integrated frameworks, which serve as the essential infrastructure, have given rise to many LLM-integrated web apps. However, some of these frameworks suffer from Remote Code Execution (RCE) vulnerabilities, allowing attackers to execute arbitrary code on apps' servers remotely via prompt injections. Despite the severity of these vulnerabilities, no existing work has been conducted for a systematic investigation of them. This leaves a great challenge on how to detect vulnerabilities in frameworks as well as LLM-integrated apps in real-world scenarios.   To fill this gap, we present two novel strategies, including 1) a static analysis-based tool called LLMSmith to scan the source code of the framework to detect potential RCE vulnerabilities and 2) a prompt-based automated testing approach to verify the vulnerability in LLM-integrated web apps. We discovered 13 vulnerabilities in 6 frameworks, including 12 RCE vulnerabilities and 1 arbitrary file read/write vulnerability. 11 of them are confirmed by the framework developers, resulting in the assignment of 7 CVE IDs. After testing 51 apps, we found vulnerabilities in 17 apps, 16 of which are vulnerable to RCE and 1 to SQL injection. We responsibly reported all 17 issues to the corresponding developers and received acknowledgments. Furthermore, we amplify the attack impact beyond achieving RCE by allowing attackers to exploit other app users (e.g. app responses hijacking, user API key leakage) without direct interaction between the attacker and the victim. Lastly, we propose some mitigating strategies for improving the security awareness of both framework and app developers, helping them to mitigate these risks effectively.



## **39. A Comprehensive Overview of Backdoor Attacks in Large Language Models within Communication Networks**

cs.CR

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2308.14367v2) [paper-pdf](http://arxiv.org/pdf/2308.14367v2)

**Authors**: Haomiao Yang, Kunlan Xiang, Mengyu Ge, Hongwei Li, Rongxing Lu, Shui Yu

**Abstract**: The Large Language Models (LLMs) are poised to offer efficient and intelligent services for future mobile communication networks, owing to their exceptional capabilities in language comprehension and generation. However, the extremely high data and computational resource requirements for the performance of LLMs compel developers to resort to outsourcing training or utilizing third-party data and computing resources. These strategies may expose the model within the network to maliciously manipulated training data and processing, providing an opportunity for attackers to embed a hidden backdoor into the model, termed a backdoor attack. Backdoor attack in LLMs refers to embedding a hidden backdoor in LLMs that causes the model to perform normally on benign samples but exhibit degraded performance on poisoned ones. This issue is particularly concerning within communication networks where reliability and security are paramount. Despite the extensive research on backdoor attacks, there remains a lack of in-depth exploration specifically within the context of LLMs employed in communication networks, and a systematic review of such attacks is currently absent. In this survey, we systematically propose a taxonomy of backdoor attacks in LLMs as used in communication networks, dividing them into four major categories: input-triggered, prompt-triggered, instruction-triggered, and demonstration-triggered attacks. Furthermore, we conduct a comprehensive analysis of the benchmark datasets. Finally, we identify potential problems and open challenges, offering valuable insights into future research directions for enhancing the security and integrity of LLMs in communication networks.



## **40. Certifying LLM Safety against Adversarial Prompting**

cs.CL

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.02705v1) [paper-pdf](http://arxiv.org/pdf/2309.02705v1)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Soheil Feizi, Hima Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial prompts, which contain maliciously designed token sequences to circumvent the model's safety guards and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We erase tokens individually and inspect the resulting subsequences using a safety filter. Our procedure labels the input prompt as harmful if any subsequences or the input prompt are detected as harmful by the filter. This guarantees that any adversarial modification of a harmful prompt up to a certain size is also labeled harmful. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Empirical results demonstrate that our technique obtains strong certified safety guarantees on harmful prompts while maintaining good performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 93% of the harmful prompts and labels 94% of the safe prompts as safe using the open source language model Llama 2 as the safety filter.



## **41. Baseline Defenses for Adversarial Attacks Against Aligned Language Models**

cs.LG

12 pages

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.00614v2) [paper-pdf](http://arxiv.org/pdf/2309.00614v2)

**Authors**: Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein

**Abstract**: As Large Language Models quickly become ubiquitous, it becomes critical to understand their security vulnerabilities. Recent work shows that text optimizers can produce jailbreaking prompts that bypass moderation and alignment. Drawing from the rich body of work on adversarial machine learning, we approach these attacks with three questions: What threat models are practically useful in this domain? How do baseline defense techniques perform in this new domain? How does LLM security differ from computer vision?   We evaluate several baseline defense strategies against leading adversarial attacks on LLMs, discussing the various settings in which each is feasible and effective. Particularly, we look at three types of defenses: detection (perplexity based), input preprocessing (paraphrase and retokenization), and adversarial training. We discuss white-box and gray-box settings and discuss the robustness-performance trade-off for each of the defenses considered. We find that the weakness of existing discrete optimizers for text, combined with the relatively high costs of optimization, makes standard adaptive attacks more challenging for LLMs. Future research will be needed to uncover whether more powerful optimizers can be developed, or whether the strength of filtering and preprocessing defenses is greater in the LLMs domain than it has been in computer vision.



## **42. MathAttack: Attacking Large Language Models Towards Math Solving Ability**

cs.CL

11 pages, 6 figures

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01686v1) [paper-pdf](http://arxiv.org/pdf/2309.01686v1)

**Authors**: Zihao Zhou, Qiufeng Wang, Mingyu Jin, Jie Yao, Jianan Ye, Wei Liu, Wei Wang, Xiaowei Huang, Kaizhu Huang

**Abstract**: With the boom of Large Language Models (LLMs), the research of solving Math Word Problem (MWP) has recently made great progress. However, there are few studies to examine the security of LLMs in math solving ability. Instead of attacking prompts in the use of LLMs, we propose a MathAttack model to attack MWP samples which are closer to the essence of security in solving math problems. Compared to traditional text adversarial attack, it is essential to preserve the mathematical logic of original MWPs during the attacking. To this end, we propose logical entity recognition to identify logical entries which are then frozen. Subsequently, the remaining text are attacked by adopting a word-level attacker. Furthermore, we propose a new dataset RobustMath to evaluate the robustness of LLMs in math solving ability. Extensive experiments on our RobustMath and two another math benchmark datasets GSM8K and MultiAirth show that MathAttack could effectively attack the math solving ability of LLMs. In the experiments, we observe that (1) Our adversarial samples from higher-accuracy LLMs are also effective for attacking LLMs with lower accuracy (e.g., transfer from larger to smaller-size LLMs, or from few-shot to zero-shot prompts); (2) Complex MWPs (such as more solving steps, longer text, more numbers) are more vulnerable to attack; (3) We can improve the robustness of LLMs by using our adversarial samples in few-shot prompts. Finally, we hope our practice and observation can serve as an important attempt towards enhancing the robustness of LLMs in math solving ability. We will release our code and dataset.



## **43. OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples**

cs.CL

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2307.11729v2) [paper-pdf](http://arxiv.org/pdf/2307.11729v2)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points in F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points in F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.



## **44. Combing for Credentials: Active Pattern Extraction from Smart Reply**

cs.CR

**SubmitDate**: 2023-09-02    [abs](http://arxiv.org/abs/2207.10802v3) [paper-pdf](http://arxiv.org/pdf/2207.10802v3)

**Authors**: Bargav Jayaraman, Esha Ghosh, Melissa Chase, Sambuddha Roy, Wei Dai, David Evans

**Abstract**: Pre-trained large language models, such as GPT\nobreakdash-2 and BERT, are often fine-tuned to achieve state-of-the-art performance on a downstream task. One natural example is the ``Smart Reply'' application where a pre-trained model is tuned to provide suggested responses for a given query message. Since the tuning data is often sensitive data such as emails or chat transcripts, it is important to understand and mitigate the risk that the model leaks its tuning data. We investigate potential information leakage vulnerabilities in a typical Smart Reply pipeline. We consider a realistic setting where the adversary can only interact with the underlying model through a front-end interface that constrains what types of queries can be sent to the model. Previous attacks do not work in these settings, but require the ability to send unconstrained queries directly to the model. Even when there are no constraints on the queries, previous attacks typically require thousands, or even millions, of queries to extract useful information, while our attacks can extract sensitive data in just a handful of queries. We introduce a new type of active extraction attack that exploits canonical patterns in text containing sensitive data. We show experimentally that it is possible for an adversary to extract sensitive user information present in the training data, even in realistic settings where all interactions with the model must go through a front-end that limits the types of queries. We explore potential mitigation strategies and demonstrate empirically how differential privacy appears to be a reasonably effective defense mechanism to such pattern extraction attacks.



## **45. Why do universal adversarial attacks work on large language models?: Geometry might be the answer**

cs.LG

2nd AdvML Frontiers Workshop at 40th International Conference on  Machine Learning, Honolulu, Hawaii, USA, 2023

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2309.00254v1) [paper-pdf](http://arxiv.org/pdf/2309.00254v1)

**Authors**: Varshini Subhash, Anna Bialas, Weiwei Pan, Finale Doshi-Velez

**Abstract**: Transformer based large language models with emergent capabilities are becoming increasingly ubiquitous in society. However, the task of understanding and interpreting their internal workings, in the context of adversarial attacks, remains largely unsolved. Gradient-based universal adversarial attacks have been shown to be highly effective on large language models and potentially dangerous due to their input-agnostic nature. This work presents a novel geometric perspective explaining universal adversarial attacks on large language models. By attacking the 117M parameter GPT-2 model, we find evidence indicating that universal adversarial triggers could be embedding vectors which merely approximate the semantic information in their adversarial training region. This hypothesis is supported by white-box model analysis comprising dimensionality reduction and similarity measurement of hidden representations. We believe this new geometric perspective on the underlying mechanism driving universal attacks could help us gain deeper insight into the internal workings and failure modes of LLMs, thus enabling their mitigation.



## **46. Temporal-Distributed Backdoor Attack Against Video Based Action Recognition**

cs.CV

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2308.11070v2) [paper-pdf](http://arxiv.org/pdf/2308.11070v2)

**Authors**: Xi Li, Songhe Wang, Ruiquan Huang, Mahanth Gowda, George Kesidis

**Abstract**: Deep neural networks (DNNs) have achieved tremendous success in various applications including video action recognition, yet remain vulnerable to backdoor attacks (Trojans). The backdoor-compromised model will mis-classify to the target class chosen by the attacker when a test instance (from a non-target class) is embedded with a specific trigger, while maintaining high accuracy on attack-free instances. Although there are extensive studies on backdoor attacks against image data, the susceptibility of video-based systems under backdoor attacks remains largely unexplored. Current studies are direct extensions of approaches proposed for image data, e.g., the triggers are independently embedded within the frames, which tend to be detectable by existing defenses. In this paper, we introduce a simple yet effective backdoor attack against video data. Our proposed attack, adding perturbations in a transformed domain, plants an imperceptible, temporally distributed trigger across the video frames, and is shown to be resilient to existing defensive strategies. The effectiveness of the proposed attack is demonstrated by extensive experiments with various well-known models on two video recognition benchmarks, UCF101 and HMDB51, and a sign language recognition benchmark, Greek Sign Language (GSL) dataset. We delve into the impact of several influential factors on our proposed attack and identify an intriguing effect termed "collateral damage" through extensive studies.



## **47. LLM in the Shell: Generative Honeypots**

cs.CR

5 pages. 1 figure 1 table

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2309.00155v1) [paper-pdf](http://arxiv.org/pdf/2309.00155v1)

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia

**Abstract**: Honeypots are essential tools in cybersecurity. However, most of them (even the high-interaction ones) lack the required realism to engage and fool human attackers. This limitation makes them easily discernible, hindering their effectiveness. This work introduces a novel method to create dynamic and realistic software honeypots based on Large Language Models. Preliminary results indicate that LLMs can create credible and dynamic honeypots capable of addressing important limitations of previous honeypots, such as deterministic responses, lack of adaptability, etc. We evaluated the realism of each command by conducting an experiment with human attackers who needed to say if the answer from the honeypot was fake or not. Our proposed honeypot, called shelLM, reached an accuracy rate of 0.92.



## **48. The Effectiveness of Large Language Models (ChatGPT and CodeBERT) for Security-Oriented Code Analysis**

cs.CR

3 Table, 8 figures

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2307.12488v3) [paper-pdf](http://arxiv.org/pdf/2307.12488v3)

**Authors**: Zhilong Wang, Lan Zhang, Chen Cao, Peng Liu

**Abstract**: Large Language Models (LLMs), such as GPT and BERT, have demonstrated remarkable capabilities in addressing neural language process tasks. Recently, the release of ChatGPT has garnered significant attention due to its ability to analyze, comprehend, and synthesize information from user inputs. Therefore, these LLMs were adopted by researchers in many different domains. In the realm of code analysis, researchers have applied LLMs to tasks like code review and code generation. However, we observed that the strengths and limitations of adopting these LLMs to the code analysis have not been investigated. In this paper, we delve into LLMs' capabilities in security-oriented program analysis, considering perspectives from both attackers and security analysts. We focus on two representative LLMs, ChatGPT and CodeBert, and evaluate their performance in solving typical analytic tasks with varying levels of difficulty. Given the different natures of ChatGPT and CodeBERT, we conduct a qualitative analysis of the model's output for ChatGPT and a quantitative analysis for CodeBERT, respectively. For ChatGPT, we present a case study involving several security-oriented program analysis tasks while deliberately introducing challenges to assess its responses. On the other hand, for CodeBERT, we systematically analyze and classify the features in code, quantitatively evaluating the impact of these features on the model's performance. Our study demonstrates the LLM's efficiency in learning high-level semantics from code, positioning ChatGPT as a potential asset in security-oriented contexts. However, it is essential to acknowledge certain limitations, such as the heavy reliance on well-defined variable and function names, making them unable to learn from anonymized code. We hope that our findings and analysis will offer valuable insights for future researchers in this domain.



## **49. Identifying and Mitigating the Security Risks of Generative AI**

cs.AI

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14840v1) [paper-pdf](http://arxiv.org/pdf/2308.14840v1)

**Authors**: Clark Barrett, Brad Boyd, Ellie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.



## **50. Out of the Cage: How Stochastic Parrots Win in Cyber Security Environments**

cs.CR

Under review. 10 pages plus appendices, 7 figures, 4 tables. Edit:  fix e-mails and code repository

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.12086v2) [paper-pdf](http://arxiv.org/pdf/2308.12086v2)

**Authors**: Maria Rigaki, Ondřej Lukáš, Carlos A. Catania, Sebastian Garcia

**Abstract**: Large Language Models (LLMs) have gained widespread popularity across diverse domains involving text generation, summarization, and various natural language processing tasks. Despite their inherent limitations, LLM-based designs have shown promising capabilities in planning and navigating open-world scenarios. This paper introduces a novel application of pre-trained LLMs as agents within cybersecurity network environments, focusing on their utility for sequential decision-making processes.   We present an approach wherein pre-trained LLMs are leveraged as attacking agents in two reinforcement learning environments. Our proposed agents demonstrate similar or better performance against state-of-the-art agents trained for thousands of episodes in most scenarios and configurations. In addition, the best LLM agents perform similarly to human testers of the environment without any additional training process. This design highlights the potential of LLMs to efficiently address complex decision-making tasks within cybersecurity.   Furthermore, we introduce a new network security environment named NetSecGame. The environment is designed to eventually support complex multi-agent scenarios within the network security domain. The proposed environment mimics real network attacks and is designed to be highly modular and adaptable for various scenarios.



