# Latest Large Language Model Attack Papers
**update at 2024-08-27 18:56:47**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Investigating the Effectiveness of Bayesian Spam Filters in Detecting LLM-modified Spam Mails**

cs.CR

EAI International Conference on Digital Forensics & Cyber Crime 2024

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14293v1) [paper-pdf](http://arxiv.org/pdf/2408.14293v1)

**Authors**: Malte Josten, Torben Weis

**Abstract**: Spam and phishing remain critical threats in cybersecurity, responsible for nearly 90% of security incidents. As these attacks grow in sophistication, the need for robust defensive mechanisms intensifies. Bayesian spam filters, like the widely adopted open-source SpamAssassin, are essential tools in this fight. However, the emergence of large language models (LLMs) such as ChatGPT presents new challenges. These models are not only powerful and accessible, but also inexpensive to use, raising concerns about their misuse in crafting sophisticated spam emails that evade traditional spam filters. This work aims to evaluate the robustness and effectiveness of SpamAssassin against LLM-modified email content. We developed a pipeline to test this vulnerability. Our pipeline modifies spam emails using GPT-3.5 Turbo and assesses SpamAssassin's ability to classify these modified emails correctly. The results show that SpamAssassin misclassified up to 73.7% of LLM-modified spam emails as legitimate. In contrast, a simpler dictionary-replacement attack showed a maximum success rate of only 0.4%. These findings highlight the significant threat posed by LLM-modified spam, especially given the cost-efficiency of such attacks (0.17 cents per email). This paper provides crucial insights into the vulnerabilities of current spam filters and the need for continuous improvement in cybersecurity measures.



## **2. Beyond Detection: Leveraging Large Language Models for Cyber Attack Prediction in IoT Networks**

cs.CR

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14045v1) [paper-pdf](http://arxiv.org/pdf/2408.14045v1)

**Authors**: Alaeddine Diaf, Abdelaziz Amara Korba, Nour Elislem Karabadji, Yacine Ghamri-Doudane

**Abstract**: In recent years, numerous large-scale cyberattacks have exploited Internet of Things (IoT) devices, a phenomenon that is expected to escalate with the continuing proliferation of IoT technology. Despite considerable efforts in attack detection, intrusion detection systems remain mostly reactive, responding to specific patterns or observed anomalies. This work proposes a proactive approach to anticipate and mitigate malicious activities before they cause damage. This paper proposes a novel network intrusion prediction framework that combines Large Language Models (LLMs) with Long Short Term Memory (LSTM) networks. The framework incorporates two LLMs in a feedback loop: a fine-tuned Generative Pre-trained Transformer (GPT) model for predicting network traffic and a fine-tuned Bidirectional Encoder Representations from Transformers (BERT) for evaluating the predicted traffic. The LSTM classifier model then identifies malicious packets among these predictions. Our framework, evaluated on the CICIoT2023 IoT attack dataset, demonstrates a significant improvement in predictive capabilities, achieving an overall accuracy of 98%, offering a robust solution to IoT cybersecurity challenges.



## **3. Probing the Safety Response Boundary of Large Language Models via Unsafe Decoding Path Generation**

cs.CR

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.10668v3) [paper-pdf](http://arxiv.org/pdf/2408.10668v3)

**Authors**: Haoyu Wang, Bingzhe Wu, Yatao Bian, Yongzhe Chang, Xueqian Wang, Peilin Zhao

**Abstract**: Large Language Models (LLMs) are implicit troublemakers. While they provide valuable insights and assist in problem-solving, they can also potentially serve as a resource for malicious activities. Implementing safety alignment could mitigate the risk of LLMs generating harmful responses. We argue that: even when an LLM appears to successfully block harmful queries, there may still be hidden vulnerabilities that could act as ticking time bombs. To identify these underlying weaknesses, we propose to use a cost value model as both a detector and an attacker. Trained on external or self-generated harmful datasets, the cost value model could successfully influence the original safe LLM to output toxic content in decoding process. For instance, LLaMA-2-chat 7B outputs 39.18% concrete toxic content, along with only 22.16% refusals without any harmful suffixes. These potential weaknesses can then be exploited via prompt optimization such as soft prompts on images. We name this decoding strategy: Jailbreak Value Decoding (JVD), emphasizing that seemingly secure LLMs may not be as safe as we initially believe. They could be used to gather harmful data or launch covert attacks.



## **4. TF-Attack: Transferable and Fast Adversarial Attacks on Large Language Models**

cs.CL

14 pages, 6 figures. arXiv admin note: text overlap with  arXiv:2305.17440 by other authors

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.13985v1) [paper-pdf](http://arxiv.org/pdf/2408.13985v1)

**Authors**: Zelin Li, Kehai Chen, Xuefeng Bai, Lemao Liu, Mingming Yang, Yang Xiang, Min Zhang

**Abstract**: With the great advancements in large language models (LLMs), adversarial attacks against LLMs have recently attracted increasing attention. We found that pre-existing adversarial attack methodologies exhibit limited transferability and are notably inefficient, particularly when applied to LLMs. In this paper, we analyze the core mechanisms of previous predominant adversarial attack methods, revealing that 1) the distributions of importance score differ markedly among victim models, restricting the transferability; 2) the sequential attack processes induces substantial time overheads. Based on the above two insights, we introduce a new scheme, named TF-Attack, for Transferable and Fast adversarial attacks on LLMs. TF-Attack employs an external LLM as a third-party overseer rather than the victim model to identify critical units within sentences. Moreover, TF-Attack introduces the concept of Importance Level, which allows for parallel substitutions of attacks. We conduct extensive experiments on 6 widely adopted benchmarks, evaluating the proposed method through both automatic and human metrics. Results show that our method consistently surpasses previous methods in transferability and delivers significant speed improvements, up to 20 times faster than earlier attack strategies.



## **5. Large Language Models as Carriers of Hidden Messages**

cs.CL

Work in progress. Code is available at  https://github.com/j-hoscilowic/zurek-stegano

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2406.02481v3) [paper-pdf](http://arxiv.org/pdf/2406.02481v3)

**Authors**: Jakub Hoscilowicz, Pawel Popiolek, Jan Rudkowski, Jedrzej Bieniasz, Artur Janicki

**Abstract**: With the help of simple fine-tuning, one can artificially embed hidden text into large language models (LLMs). This text is revealed only when triggered by a specific query to the LLM. Two primary applications are LLM fingerprinting and steganography. In the context of LLM fingerprinting, a unique text identifier (fingerprint) is embedded within the model to verify licensing compliance. In the context of steganography, the LLM serves as a carrier for hidden messages that can be disclosed through a chosen trigger question.   Our work demonstrates that embedding hidden text in the LLM via fine-tuning, though seemingly secure due to the vast number of potential triggers (any sequence of characters or tokens could serve as a trigger), is susceptible to extraction through analysis of the LLM's output decoding process. We propose an extraction attack called Unconditional Token Forcing (UTF). It is premised on the hypothesis that iteratively feeding each token from the LLM's vocabulary into the model should reveal output sequences with abnormally high token probabilities, indicating potential hidden text candidates. We also present a defense method to hide text in such a way that it is resistant to both UTF and attacks based on sampling decoding methods, which we named Unconditional Token Forcing Confusion (UTFC). To the best of our knowledge, there is no attack method that can extract text hidden with UTFC. UTFC has both benign applications (improving LLM fingerprinting) and malign applications (using LLMs to create covert communication channels).



## **6. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2403.17710v2) [paper-pdf](http://arxiv.org/pdf/2403.17710v2)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies.



## **7. Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors**

cs.CV

15 pages

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2405.10529v2) [paper-pdf](http://arxiv.org/pdf/2405.10529v2)

**Authors**: Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, Chaowei Xiao

**Abstract**: Large language models have become increasingly prominent, also signaling a shift towards multimodality as the next frontier in artificial intelligence, where their embeddings are harnessed as prompts to generate textual content. Vision-language models (VLMs) stand at the forefront of this advancement, offering innovative ways to combine visual and textual data for enhanced understanding and interaction. However, this integration also enlarges the attack surface. Patch-based adversarial attack is considered the most realistic threat model in physical vision applications, as demonstrated in many existing literature. In this paper, we propose to address patched visual prompt injection, where adversaries exploit adversarial patches to generate target content in VLMs. Our investigation reveals that patched adversarial prompts exhibit sensitivity to pixel-wise randomization, a trait that remains robust even against adaptive attacks designed to counteract such defenses. Leveraging this insight, we introduce SmoothVLM, a defense mechanism rooted in smoothing techniques, specifically tailored to protect VLMs from the threat of patched visual prompt injectors. Our framework significantly lowers the attack success rate to a range between 0% and 5.0% on two leading VLMs, while achieving around 67.3% to 95.0% context recovery of the benign images, demonstrating a balance between security and usability.



## **8. Probing the Robustness of Vision-Language Pretrained Models: A Multimodal Adversarial Attack Approach**

cs.CV

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2408.13461v1) [paper-pdf](http://arxiv.org/pdf/2408.13461v1)

**Authors**: Jiwei Guan, Tianyu Ding, Longbing Cao, Lei Pan, Chen Wang, Xi Zheng

**Abstract**: Vision-language pretraining (VLP) with transformers has demonstrated exceptional performance across numerous multimodal tasks. However, the adversarial robustness of these models has not been thoroughly investigated. Existing multimodal attack methods have largely overlooked cross-modal interactions between visual and textual modalities, particularly in the context of cross-attention mechanisms. In this paper, we study the adversarial vulnerability of recent VLP transformers and design a novel Joint Multimodal Transformer Feature Attack (JMTFA) that concurrently introduces adversarial perturbations in both visual and textual modalities under white-box settings. JMTFA strategically targets attention relevance scores to disrupt important features within each modality, generating adversarial samples by fusing perturbations and leading to erroneous model predictions. Experimental results indicate that the proposed approach achieves high attack success rates on vision-language understanding and reasoning downstream tasks compared to existing baselines. Notably, our findings reveal that the textual modality significantly influences the complex fusion processes within VLP transformers. Moreover, we observe no apparent relationship between model size and adversarial robustness under our proposed attacks. These insights emphasize a new dimension of adversarial robustness and underscore potential risks in the reliable deployment of multimodal AI systems.



## **9. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

cs.LG

END (will never be modified again!!) :Jumps-Diffusion and stock  market: Better quantify uncertainty in financial simulations

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2407.14573v4) [paper-pdf](http://arxiv.org/pdf/2407.14573v4)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.



## **10. Is Generative AI the Next Tactical Cyber Weapon For Threat Actors? Unforeseen Implications of AI Generated Cyber Attacks**

cs.CR

Journal Paper

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.12806v1) [paper-pdf](http://arxiv.org/pdf/2408.12806v1)

**Authors**: Yusuf Usman, Aadesh Upadhyay, Prashnna Gyawali, Robin Chataut

**Abstract**: In an era where digital threats are increasingly sophisticated, the intersection of Artificial Intelligence and cybersecurity presents both promising defenses and potent dangers. This paper delves into the escalating threat posed by the misuse of AI, specifically through the use of Large Language Models (LLMs). This study details various techniques like the switch method and character play method, which can be exploited by cybercriminals to generate and automate cyber attacks. Through a series of controlled experiments, the paper demonstrates how these models can be manipulated to bypass ethical and privacy safeguards to effectively generate cyber attacks such as social engineering, malicious code, payload generation, and spyware. By testing these AI generated attacks on live systems, the study assesses their effectiveness and the vulnerabilities they exploit, offering a practical perspective on the risks AI poses to critical infrastructure. We also introduce Occupy AI, a customized, finetuned LLM specifically engineered to automate and execute cyberattacks. This specialized AI driven tool is adept at crafting steps and generating executable code for a variety of cyber threats, including phishing, malware injection, and system exploitation. The results underscore the urgency for ethical AI practices, robust cybersecurity measures, and regulatory oversight to mitigate AI related threats. This paper aims to elevate awareness within the cybersecurity community about the evolving digital threat landscape, advocating for proactive defense strategies and responsible AI development to protect against emerging cyber threats.



## **11. BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models**

cs.AI

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.12798v1) [paper-pdf](http://arxiv.org/pdf/2408.12798v1)

**Authors**: Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, Jun Sun

**Abstract**: Generative Large Language Models (LLMs) have made significant strides across various tasks, but they remain vulnerable to backdoor attacks, where specific triggers in the prompt cause the LLM to generate adversary-desired responses. While most backdoor research has focused on vision or text classification tasks, backdoor attacks in text generation have been largely overlooked. In this work, we introduce \textit{BackdoorLLM}, the first comprehensive benchmark for studying backdoor attacks on LLMs. \textit{BackdoorLLM} features: 1) a repository of backdoor benchmarks with a standardized training pipeline, 2) diverse attack strategies, including data poisoning, weight poisoning, hidden state attacks, and chain-of-thought attacks, 3) extensive evaluations with over 200 experiments on 8 attacks across 7 scenarios and 6 model architectures, and 4) key insights into the effectiveness and limitations of backdoors in LLMs. We hope \textit{BackdoorLLM} will raise awareness of backdoor threats and contribute to advancing AI safety. The code is available at \url{https://github.com/bboylyg/BackdoorLLM}.



## **12. Can Large Language Models Automatically Jailbreak GPT-4V?**

cs.CL

TrustNLP@NAACL2024 (Fourth Workshop on Trustworthy Natural Language  Processing)

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2407.16686v2) [paper-pdf](http://arxiv.org/pdf/2407.16686v2)

**Authors**: Yuanwei Wu, Yue Huang, Yixin Liu, Xiang Li, Pan Zhou, Lichao Sun

**Abstract**: GPT-4V has attracted considerable attention due to its extraordinary capacity for integrating and processing multimodal information. At the same time, its ability of face recognition raises new safety concerns of privacy leakage. Despite researchers' efforts in safety alignment through RLHF or preprocessing filters, vulnerabilities might still be exploited. In our study, we introduce AutoJailbreak, an innovative automatic jailbreak technique inspired by prompt optimization. We leverage Large Language Models (LLMs) for red-teaming to refine the jailbreak prompt and employ weak-to-strong in-context learning prompts to boost efficiency. Furthermore, we present an effective search method that incorporates early stopping to minimize optimization time and token expenditure. Our experiments demonstrate that AutoJailbreak significantly surpasses conventional methods, achieving an Attack Success Rate (ASR) exceeding 95.3\%. This research sheds light on strengthening GPT-4V security, underscoring the potential for LLMs to be exploited in compromising GPT-4V integrity.



## **13. LLM-PBE: Assessing Data Privacy in Large Language Models**

cs.CR

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.12787v1) [paper-pdf](http://arxiv.org/pdf/2408.12787v1)

**Authors**: Qinbin Li, Junyuan Hong, Chulin Xie, Jeffrey Tan, Rachel Xin, Junyi Hou, Xavier Yin, Zhun Wang, Dan Hendrycks, Zhangyang Wang, Bo Li, Bingsheng He, Dawn Song

**Abstract**: Large Language Models (LLMs) have become integral to numerous domains, significantly advancing applications in data management, mining, and analysis. Their profound capabilities in processing and interpreting complex language data, however, bring to light pressing concerns regarding data privacy, especially the risk of unintentional training data leakage. Despite the critical nature of this issue, there has been no existing literature to offer a comprehensive assessment of data privacy risks in LLMs. Addressing this gap, our paper introduces LLM-PBE, a toolkit crafted specifically for the systematic evaluation of data privacy risks in LLMs. LLM-PBE is designed to analyze privacy across the entire lifecycle of LLMs, incorporating diverse attack and defense strategies, and handling various data types and metrics. Through detailed experimentation with multiple LLMs, LLM-PBE facilitates an in-depth exploration of data privacy concerns, shedding light on influential factors such as model size, data characteristics, and evolving temporal dimensions. This study not only enriches the understanding of privacy issues in LLMs but also serves as a vital resource for future research in the field. Aimed at enhancing the breadth of knowledge in this area, the findings, resources, and our full technical report are made available at https://llm-pbe.github.io/, providing an open platform for academic and practical advancements in LLM privacy assessment.



## **14. Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.08924v2) [paper-pdf](http://arxiv.org/pdf/2408.08924v2)

**Authors**: Jiawei Zhao, Kejiang Chen, Xiaojian Yuan, Weiming Zhang

**Abstract**: In recent years, the rapid development of large language models (LLMs) has achieved remarkable performance across various tasks. However, research indicates that LLMs are vulnerable to jailbreak attacks, where adversaries can induce the generation of harmful content through meticulously crafted prompts. This vulnerability poses significant challenges to the secure use and promotion of LLMs. Existing defense methods offer protection from different perspectives but often suffer from insufficient effectiveness or a significant impact on the model's capabilities. In this paper, we propose a plug-and-play and easy-to-deploy jailbreak defense framework, namely Prefix Guidance (PG), which guides the model to identify harmful prompts by directly setting the first few tokens of the model's output. This approach combines the model's inherent security capabilities with an external classifier to defend against jailbreak attacks. We demonstrate the effectiveness of PG across three models and five attack methods. Compared to baselines, our approach is generally more effective on average. Additionally, results on the Just-Eval benchmark further confirm PG's superiority to preserve the model's performance. our code is available at https://github.com/weiyezhimeng/Prefix-Guidance.



## **15. The Dark Side of Function Calling: Pathways to Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2407.17915v2) [paper-pdf](http://arxiv.org/pdf/2407.17915v2)

**Authors**: Zihui Wu, Haichang Gao, Jianping He, Ping Wang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but their power comes with significant security considerations. While extensive research has been conducted on the safety of LLMs in chat mode, the security implications of their function calling feature have been largely overlooked. This paper uncovers a critical vulnerability in the function calling process of LLMs, introducing a novel "jailbreak function" attack method that exploits alignment discrepancies, user coercion, and the absence of rigorous safety filters. Our empirical study, conducted on six state-of-the-art LLMs including GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-pro, reveals an alarming average success rate of over 90\% for this attack. We provide a comprehensive analysis of why function calls are susceptible to such attacks and propose defensive strategies, including the use of defensive prompts. Our findings highlight the urgent need for enhanced security measures in the function calling capabilities of LLMs, contributing to the field of AI safety by identifying a previously unexplored risk, designing an effective attack method, and suggesting practical defensive measures. Our code is available at https://github.com/wooozihui/jailbreakfunction.



## **16. Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning**

cs.LG

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2402.01109v4) [paper-pdf](http://arxiv.org/pdf/2402.01109v4)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.



## **17. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2407.15549v2) [paper-pdf](http://arxiv.org/pdf/2407.15549v2)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.



## **18. A Study of Backdoors in Instruction Fine-tuned Language Models**

cs.CR

Under review

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.07778v2) [paper-pdf](http://arxiv.org/pdf/2406.07778v2)

**Authors**: Jayaram Raghuram, George Kesidis, David J. Miller

**Abstract**: Backdoor data poisoning, inserted within instruction examples used to fine-tune a foundation Large Language Model (LLM) for downstream tasks (\textit{e.g.,} sentiment prediction), is a serious security concern due to the evasive nature of such attacks. The poisoning is usually in the form of a (seemingly innocuous) trigger word or phrase inserted into a very small fraction of the fine-tuning samples from a target class. Such backdoor attacks can: alter response sentiment, violate censorship, over-refuse (invoke censorship for legitimate queries), inject false content, or trigger nonsense responses (hallucinations). In this work we investigate the efficacy of instruction fine-tuning backdoor attacks as attack "hyperparameters" are varied under a variety of scenarios, considering: the trigger location in the poisoned examples; robustness to change in the trigger location, partial triggers, and synonym substitutions at test time; attack transfer from one (fine-tuning) domain to a related test domain; and clean-label vs. dirty-label poisoning. Based on our observations, we propose and evaluate two defenses against these attacks: i) a \textit{during-fine-tuning defense} based on word-frequency counts that assumes the (possibly poisoned) fine-tuning dataset is available and identifies the backdoor trigger tokens; and ii) a \textit{post-fine-tuning defense} based on downstream clean fine-tuning of the backdoored LLM with a small defense dataset. Finally, we provide a brief survey of related work on backdoor attacks and defenses.



## **19. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2402.06255v3) [paper-pdf](http://arxiv.org/pdf/2402.06255v3)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreak attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly with a particular focus on harmful content filtering or heuristical defensive prompt designs. However, how to achieve intrinsic robustness through the prompts remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both grey-box and black-box attacks, reducing the success rate of advanced attacks to nearly 0 while maintaining the model's utility on the benign task. The proposed defense strategy incurs only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/rain152/PAT.



## **20. Competence-Based Analysis of Language Models**

cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2303.00333v4) [paper-pdf](http://arxiv.org/pdf/2303.00333v4)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain and predict behaviors across these tasks.



## **21. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11749v1) [paper-pdf](http://arxiv.org/pdf/2408.11749v1)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.



## **22. Watch Out for Your Guidance on Generation! Exploring Conditional Backdoor Attacks against Large Language Models**

cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2404.14795v4) [paper-pdf](http://arxiv.org/pdf/2404.14795v4)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream backdoor attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of backdoor activation, we present a new poisoning paradigm against LLMs triggered by specifying generation conditions, which are commonly adopted strategies by users during model inference. The poisoned model performs normally for output under normal/other generation conditions, while becomes harmful for output under target generation conditions. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation conditions by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our attack can be generally divided into two types with different targets: Safety unalignment attack and Ability degradation attack. Our extensive experiments demonstrate that BrieFool is effective across safety domains and ability domains, achieving higher success rates than baseline methods, with 94.3 % on GPT-3.5-turbo



## **23. Large Language Models are Good Attackers: Efficient and Stealthy Textual Backdoor Attacks**

cs.CL

Under Review

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11587v1) [paper-pdf](http://arxiv.org/pdf/2408.11587v1)

**Authors**: Ziqiang Li, Yueqi Zeng, Pengfei Xia, Lei Liu, Zhangjie Fu, Bin Li

**Abstract**: With the burgeoning advancements in the field of natural language processing (NLP), the demand for training data has increased significantly. To save costs, it has become common for users and businesses to outsource the labor-intensive task of data collection to third-party entities. Unfortunately, recent research has unveiled the inherent risk associated with this practice, particularly in exposing NLP systems to potential backdoor attacks. Specifically, these attacks enable malicious control over the behavior of a trained model by poisoning a small portion of the training data. Unlike backdoor attacks in computer vision, textual backdoor attacks impose stringent requirements for attack stealthiness. However, existing attack methods meet significant trade-off between effectiveness and stealthiness, largely due to the high information entropy inherent in textual data. In this paper, we introduce the Efficient and Stealthy Textual backdoor attack method, EST-Bad, leveraging Large Language Models (LLMs). Our EST-Bad encompasses three core strategies: optimizing the inherent flaw of models as the trigger, stealthily injecting triggers with LLMs, and meticulously selecting the most impactful samples for backdoor injection. Through the integration of these techniques, EST-Bad demonstrates an efficient achievement of competitive attack performance while maintaining superior stealthiness compared to prior methods across various text classifier datasets.



## **24. SHIELD: Evaluation and Defense Strategies for Copyright Compliance in LLM Text Generation**

cs.CL

Work in progress

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.12975v2) [paper-pdf](http://arxiv.org/pdf/2406.12975v2)

**Authors**: Xiaoze Liu, Ting Sun, Tianyang Xu, Feijie Wu, Cunxiang Wang, Xiaoqian Wang, Jing Gao

**Abstract**: Large Language Models (LLMs) have transformed machine learning but raised significant legal concerns due to their potential to produce text that infringes on copyrights, resulting in several high-profile lawsuits. The legal landscape is struggling to keep pace with these rapid advancements, with ongoing debates about whether generated text might plagiarize copyrighted materials. Current LLMs may infringe on copyrights or overly restrict non-copyrighted texts, leading to these challenges: (i) the need for a comprehensive evaluation benchmark to assess copyright compliance from multiple aspects; (ii) evaluating robustness against safeguard bypassing attacks; and (iii) developing effective defense targeted against the generation of copyrighted text. To tackle these challenges, we introduce a curated dataset to evaluate methods, test attack strategies, and propose lightweight, real-time defense to prevent the generation of copyrighted text, ensuring the safe and lawful use of LLMs. Our experiments demonstrate that current LLMs frequently output copyrighted text, and that jailbreaking attacks can significantly increase the volume of copyrighted output. Our proposed defense mechanism significantly reduces the volume of copyrighted text generated by LLMs by effectively refusing malicious requests. Code is publicly available at https://github.com/xz-liu/SHIELD



## **25. EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models**

cs.AI

19 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11308v1) [paper-pdf](http://arxiv.org/pdf/2408.11308v1)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. Built upon this idea, we introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85\% in comparison with 50\% for the present SOTAs, with minimal impact on the utility and effectiveness of LLMs.



## **26. Medical MLLM is Vulnerable: Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models**

cs.CR

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2405.20775v2) [paper-pdf](http://arxiv.org/pdf/2405.20775v2)

**Authors**: Xijie Huang, Xinyuan Wang, Hantao Zhang, Yinghao Zhu, Jiawen Xi, Jingkun An, Hao Wang, Hao Liang, Chengwei Pan

**Abstract**: Security concerns related to Large Language Models (LLMs) have been extensively explored, yet the safety implications for Multimodal Large Language Models (MLLMs), particularly in medical contexts (MedMLLMs), remain insufficiently studied. This paper delves into the underexplored security vulnerabilities of MedMLLMs, especially when deployed in clinical environments where the accuracy and relevance of question-and-answer interactions are critically tested against complex medical challenges. By combining existing clinical medical data with atypical natural phenomena, we define the mismatched malicious attack (2M-attack) and introduce its optimized version, known as the optimized mismatched malicious attack (O2M-attack or 2M-optimization). Using the voluminous 3MAD dataset that we construct, which covers a wide range of medical image modalities and harmful medical scenarios, we conduct a comprehensive analysis and propose the MCM optimization method, which significantly enhances the attack success rate on MedMLLMs. Evaluations with this dataset and attack methods, including white-box attacks on LLaVA-Med and transfer attacks (black-box) on four other SOTA models, indicate that even MedMLLMs designed with enhanced security features remain vulnerable to security breaches. Our work underscores the urgent need for a concerted effort to implement robust security measures and enhance the safety and efficacy of open-source MedMLLMs, particularly given the potential severity of jailbreak attacks and other malicious or clinically significant exploits in medical settings. Our code is available at https://github.com/dirtycomputer/O2M_attack.



## **27. Hide Your Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Neural Carrier Articles**

cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11182v1) [paper-pdf](http://arxiv.org/pdf/2408.11182v1)

**Authors**: Zhilong Wang, Haizhou Wang, Nanqing Luo, Lan Zhang, Xiaoyan Sun, Yebo Cao, Peng Liu

**Abstract**: Jailbreak attacks on Language Model Models (LLMs) entail crafting prompts aimed at exploiting the models to generate malicious content. This paper proposes a new type of jailbreak attacks which shift the attention of the LLM by inserting a prohibited query into a carrier article. The proposed attack leverage the knowledge graph and a composer LLM to automatically generating a carrier article that is similar to the topic of the prohibited query but does not violate LLM's safeguards. By inserting the malicious query to the carrier article, the assembled attack payload can successfully jailbreak LLM. To evaluate the effectiveness of our method, we leverage 4 popular categories of ``harmful behaviors'' adopted by related researches to attack 6 popular LLMs. Our experiment results show that the proposed attacking method can successfully jailbreak all the target LLMs which high success rate, except for Claude-3.



## **28. Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks**

cs.CL

Accepted to KDD 2024 (Research Track)

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2310.10830v2) [paper-pdf](http://arxiv.org/pdf/2310.10830v2)

**Authors**: Jiaying Wu, Jiafeng Guo, Bryan Hooi

**Abstract**: It is commonly perceived that fake news and real news exhibit distinct writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the advent of powerful Large Language Models (LLMs) has empowered malicious actors to mimic the style of trustworthy news sources, doing so swiftly, cost-effectively, and at scale. Our analysis reveals that LLM-camouflaged fake news content significantly undermines the effectiveness of state-of-the-art text-based detectors (up to 38% decrease in F1 Score), implying a severe vulnerability to stylistic variations. To address this, we introduce SheepDog, a style-robust fake news detector that prioritizes content over style in determining news veracity. SheepDog achieves this resilience through (1) LLM-empowered news reframings that inject style diversity into the training process by customizing articles to match different styles; (2) a style-agnostic training scheme that ensures consistent veracity predictions across style-diverse reframings; and (3) content-focused veracity attributions that distill content-centric guidelines from LLMs for debunking fake news, offering supplementary cues and potential intepretability that assist veracity prediction. Extensive experiments on three real-world benchmarks demonstrate SheepDog's style robustness and adaptability to various backbones.



## **29. While GitHub Copilot Excels at Coding, Does It Ensure Responsible Output?**

cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11006v1) [paper-pdf](http://arxiv.org/pdf/2408.11006v1)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **30. Towards Efficient Formal Verification of Spiking Neural Network**

cs.AI

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10900v1) [paper-pdf](http://arxiv.org/pdf/2408.10900v1)

**Authors**: Baekryun Seong, Jieung Kim, Sang-Ki Ko

**Abstract**: Recently, AI research has primarily focused on large language models (LLMs), and increasing accuracy often involves scaling up and consuming more power. The power consumption of AI has become a significant societal issue; in this context, spiking neural networks (SNNs) offer a promising solution. SNNs operate event-driven, like the human brain, and compress information temporally. These characteristics allow SNNs to significantly reduce power consumption compared to perceptron-based artificial neural networks (ANNs), highlighting them as a next-generation neural network technology. However, societal concerns regarding AI go beyond power consumption, with the reliability of AI models being a global issue. For instance, adversarial attacks on AI models are a well-studied problem in the context of traditional neural networks. Despite their importance, the stability and property verification of SNNs remains in the early stages of research. Most SNN verification methods are time-consuming and barely scalable, making practical applications challenging. In this paper, we introduce temporal encoding to achieve practical performance in verifying the adversarial robustness of SNNs. We conduct a theoretical analysis of this approach and demonstrate its success in verifying SNNs at previously unmanageable scales. Our contribution advances SNN verification to a practical level, facilitating the safer application of SNNs.



## **31. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10738v1) [paper-pdf](http://arxiv.org/pdf/2408.10738v1)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also encounter notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the top k relevant items from offline knowledge bases, utilizing all available information from a webpage, including logos, HTML, and URLs. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.



## **32. MEGen: Generative Backdoor in Large Language Models via Model Editing**

cs.CL

Working in progress

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10722v1) [paper-pdf](http://arxiv.org/pdf/2408.10722v1)

**Authors**: Jiyang Qiu, Xinbei Ma, Zhuosheng Zhang, Hai Zhao

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities. Their powerful generative abilities enable flexible responses based on various queries or instructions. Emerging as widely adopted generalists for diverse tasks, LLMs are still vulnerable to backdoors. This paper proposes an editing-based generative backdoor, named MEGen, aiming to create a customized backdoor for NLP tasks with the least side effects. In our approach, we first leverage a language model to insert a trigger selected on fixed metrics into the input, then design a pipeline of model editing to directly embed a backdoor into an LLM. By adjusting a small set of local parameters with a mini-batch of samples, MEGen significantly enhances time efficiency and achieves high robustness. Experimental results indicate that our backdoor attack strategy achieves a high attack success rate on poison data while maintaining the model's performance on clean data. Notably, the backdoored model, when triggered, can freely output pre-set dangerous information while successfully completing downstream tasks. This suggests that future LLM applications could be guided to deliver certain dangerous information, thus altering the LLM's generative style. We believe this approach provides insights for future LLM applications and the execution of backdoor attacks on conversational AI systems.



## **33. Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Technique**

cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10701v1) [paper-pdf](http://arxiv.org/pdf/2408.10701v1)

**Authors**: Tej Deep Pala, Vernon Y. H. Toh, Rishabh Bhardwaj, Soujanya Poria

**Abstract**: In today's era, where large language models (LLMs) are integrated into numerous real-world applications, ensuring their safety and robustness is crucial for responsible AI usage. Automated red-teaming methods play a key role in this process by generating adversarial attacks to identify and mitigate potential vulnerabilities in these models. However, existing methods often struggle with slow performance, limited categorical diversity, and high resource demands. While Rainbow Teaming, a recent approach, addresses the diversity challenge by framing adversarial prompt generation as a quality-diversity search, it remains slow and requires a large fine-tuned mutator for optimal performance. To overcome these limitations, we propose Ferret, a novel approach that builds upon Rainbow Teaming by generating multiple adversarial prompt mutations per iteration and using a scoring function to rank and select the most effective adversarial prompt. We explore various scoring functions, including reward models, Llama Guard, and LLM-as-a-judge, to rank adversarial mutations based on their potential harm to improve the efficiency of the search for harmful mutations. Our results demonstrate that Ferret, utilizing a reward model as a scoring function, improves the overall attack success rate (ASR) to 95%, which is 46% higher than Rainbow Teaming. Additionally, Ferret reduces the time needed to achieve a 90% ASR by 15.2% compared to the baseline and generates adversarial prompts that are transferable i.e. effective on other LLMs of larger size. Our codes are available at https://github.com/declare-lab/ferret.



## **34. Promoting Equality in Large Language Models: Identifying and Mitigating the Implicit Bias based on Bayesian Theory**

cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10608v1) [paper-pdf](http://arxiv.org/pdf/2408.10608v1)

**Authors**: Yongxin Deng, Xihe Qiu, Xiaoyu Tan, Jing Pan, Chen Jue, Zhijun Fang, Yinghui Xu, Wei Chu, Yuan Qi

**Abstract**: Large language models (LLMs) are trained on extensive text corpora, which inevitably include biased information. Although techniques such as Affective Alignment can mitigate some negative impacts of these biases, existing prompt-based attack methods can still extract these biases from the model's weights. Moreover, these biases frequently appear subtly when LLMs are prompted to perform identical tasks across different demographic groups, thereby camouflaging their presence. To address this issue, we have formally defined the implicit bias problem and developed an innovative framework for bias removal based on Bayesian theory, Bayesian-Theory based Bias Removal (BTBR). BTBR employs likelihood ratio screening to pinpoint data entries within publicly accessible biased datasets that represent biases inadvertently incorporated during the LLM training phase. It then automatically constructs relevant knowledge triples and expunges bias information from LLMs using model editing techniques. Through extensive experimentation, we have confirmed the presence of the implicit bias problem in LLMs and demonstrated the effectiveness of our BTBR approach.



## **35. PromptBench: A Unified Library for Evaluation of Large Language Models**

cs.AI

Accepted by Journal of Machine Learning Research (JMLR); code:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2312.07910v3) [paper-pdf](http://arxiv.org/pdf/2312.07910v3)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.



## **36. Development of an AI Anti-Bullying System Using Large Language Model Key Topic Detection**

cs.AI

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10417v1) [paper-pdf](http://arxiv.org/pdf/2408.10417v1)

**Authors**: Matthew Tassava, Cameron Kolodjski, Jordan Milbrath, Adorah Bishop, Nathan Flanders, Robbie Fetsch, Danielle Hanson, Jeremy Straub

**Abstract**: This paper presents and evaluates work on the development of an artificial intelligence (AI) anti-bullying system. The system is designed to identify coordinated bullying attacks via social media and other mechanisms, characterize them and propose remediation and response activities to them. In particular, a large language model (LLM) is used to populate an enhanced expert system-based network model of a bullying attack. This facilitates analysis and remediation activity - such as generating report messages to social media companies - determination. The system is described and the efficacy of the LLM for populating the model is analyzed herein.



## **37. A Disguised Wolf Is More Harmful Than a Toothless Tiger: Adaptive Malicious Code Injection Backdoor Attack Leveraging User Behavior as Triggers**

cs.AI

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10334v1) [paper-pdf](http://arxiv.org/pdf/2408.10334v1)

**Authors**: Shangxi Wu, Jitao Sang

**Abstract**: In recent years, large language models (LLMs) have made significant progress in the field of code generation. However, as more and more users rely on these models for software development, the security risks associated with code generation models have become increasingly significant. Studies have shown that traditional deep learning robustness issues also negatively impact the field of code generation. In this paper, we first present the game-theoretic model that focuses on security issues in code generation scenarios. This framework outlines possible scenarios and patterns where attackers could spread malicious code models to create security threats. We also pointed out for the first time that the attackers can use backdoor attacks to dynamically adjust the timing of malicious code injection, which will release varying degrees of malicious code depending on the skill level of the user. Through extensive experiments on leading code generation models, we validate our proposed game-theoretic model and highlight the significant threats that these new attack scenarios pose to the safe use of code models.



## **38. Topic-Based Watermarks for LLM-Generated Text**

cs.CR

Results for proposed scheme, additional/removal of content (figures  and equations), 12 pages

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2404.02138v3) [paper-pdf](http://arxiv.org/pdf/2404.02138v3)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: The indistinguishability of text generated by large language models (LLMs) from human-generated text poses significant challenges. Watermarking algorithms are potential solutions by embedding detectable signatures within LLM-generated outputs. However, current watermarking schemes lack robustness to a range of attacks such as text substitution or manipulation, undermining their reliability. This paper proposes a novel topic-based watermarking algorithm for LLMs, designed to enhance the robustness of watermarking in LLMs. Our approach leverages the topics extracted from input prompts or outputs of non-watermarked LLMs in the generation process of watermarked text. We dynamically utilize token lists on identified topics and adjust token sampling weights accordingly. By using these topic-specific token biases, we embed a topic-sensitive watermarking into the generated text. We outline the theoretical framework of our topic-based watermarking algorithm and discuss its potential advantages in various scenarios. Additionally, we explore a comprehensive range of attacks against watermarking algorithms, including discrete alterations, paraphrasing, and tokenizations. We demonstrate that our proposed watermarking scheme classifies various watermarked text topics with 99.99% confidence and outperforms existing algorithms in terms of z-score robustness and the feasibility of modeling text degradation by potential attackers, while considering the trade-offs between the benefits and losses of watermarking LLM-generated text.



## **39. Privacy Checklist: Privacy Violation Detection Grounding on Contextual Integrity Theory**

cs.CL

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10053v1) [paper-pdf](http://arxiv.org/pdf/2408.10053v1)

**Authors**: Haoran Li, Wei Fan, Yulin Chen, Jiayang Cheng, Tianshu Chu, Xuebing Zhou, Peizhao Hu, Yangqiu Song

**Abstract**: Privacy research has attracted wide attention as individuals worry that their private data can be easily leaked during interactions with smart devices, social platforms, and AI applications. Computer science researchers, on the other hand, commonly study privacy issues through privacy attacks and defenses on segmented fields. Privacy research is conducted on various sub-fields, including Computer Vision (CV), Natural Language Processing (NLP), and Computer Networks. Within each field, privacy has its own formulation. Though pioneering works on attacks and defenses reveal sensitive privacy issues, they are narrowly trapped and cannot fully cover people's actual privacy concerns. Consequently, the research on general and human-centric privacy research remains rather unexplored. In this paper, we formulate the privacy issue as a reasoning problem rather than simple pattern matching. We ground on the Contextual Integrity (CI) theory which posits that people's perceptions of privacy are highly correlated with the corresponding social context. Based on such an assumption, we develop the first comprehensive checklist that covers social identities, private attributes, and existing privacy regulations. Unlike prior works on CI that either cover limited expert annotated norms or model incomplete social context, our proposed privacy checklist uses the whole Health Insurance Portability and Accountability Act of 1996 (HIPAA) as an example, to show that we can resort to large language models (LLMs) to completely cover the HIPAA's regulations. Additionally, our checklist also gathers expert annotations across multiple ontologies to determine private information including but not limited to personally identifiable information (PII). We use our preliminary results on the HIPAA to shed light on future context-centric privacy research to cover more privacy regulations, social norms and standards.



## **40. Transferring Backdoors between Large Language Models by Knowledge Distillation**

cs.CR

13 pages, 16 figures, 5 tables

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09878v1) [paper-pdf](http://arxiv.org/pdf/2408.09878v1)

**Authors**: Pengzhou Cheng, Zongru Wu, Tianjie Ju, Wei Du, Zhuosheng Zhang Gongshen Liu

**Abstract**: Backdoor Attacks have been a serious vulnerability against Large Language Models (LLMs). However, previous methods only reveal such risk in specific models, or present tasks transferability after attacking the pre-trained phase. So, how risky is the model transferability of a backdoor attack? In this paper, we focus on whether existing mini-LLMs may be unconsciously instructed in backdoor knowledge by poisoned teacher LLMs through knowledge distillation (KD). Specifically, we propose ATBA, an adaptive transferable backdoor attack, which can effectively distill the backdoor of teacher LLMs into small models when only executing clean-tuning. We first propose the Target Trigger Generation (TTG) module that filters out a set of indicative trigger candidates from the token list based on cosine similarity distribution. Then, we exploit a shadow model to imitate the distilling process and introduce an Adaptive Trigger Optimization (ATO) module to realize a gradient-based greedy feedback to search optimal triggers. Extensive experiments show that ATBA generates not only positive guidance for student models but also implicitly transfers backdoor knowledge. Our attack is robust and stealthy, with over 80% backdoor transferability, and hopes the attention of security.



## **41. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

cs.CR

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2312.00029v3) [paper-pdf](http://arxiv.org/pdf/2312.00029v3)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. Such vulnerabilities can lead to LLMs being manipulated into generating hazardous content: from instructions for creating dangerous materials to inciting violence or endorsing unethical behaviors. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM acting as a guardian to the primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis reviews that by using Bergeron to complement models with existing alignment training, we can significantly improve the robustness and safety of multiple, commonly used commercial and open-source LLMs. Specifically, we found that models integrated with Bergeron are, on average, nearly seven times more resistant to attacks compared to models without such support.



## **42. Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning**

cs.AI

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09600v1) [paper-pdf](http://arxiv.org/pdf/2408.09600v1)

**Authors**: Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, Ling Liu

**Abstract**: Safety aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks \cite{qi2023fine}-- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. Existing mitigation strategies include alignment stage solutions \cite{huang2024vaccine, rosati2024representation} and fine-tuning stage solutions \cite{huang2024lazy,mukhoti2023fine}. However, our evaluation shows that both categories of defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense, which however, is necessary to guarantee finetune performance. To this end, we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can reduce harmful score while maintaining accuracy on downstream tasks.



## **43. Characterizing and Evaluating the Reliability of LLMs against Jailbreak Attacks**

cs.CL

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09326v1) [paper-pdf](http://arxiv.org/pdf/2408.09326v1)

**Authors**: Kexin Chen, Yi Liu, Dongxia Wang, Jiaying Chen, Wenhai Wang

**Abstract**: Large Language Models (LLMs) have increasingly become pivotal in content generation with notable societal impact. These models hold the potential to generate content that could be deemed harmful.Efforts to mitigate this risk include implementing safeguards to ensure LLMs adhere to social ethics.However, despite such measures, the phenomenon of "jailbreaking" -- where carefully crafted prompts elicit harmful responses from models -- persists as a significant challenge. Recognizing the continuous threat posed by jailbreaking tactics and their repercussions for the trustworthy use of LLMs, a rigorous assessment of the models' robustness against such attacks is essential. This study introduces an comprehensive evaluation framework and conducts an large-scale empirical experiment to address this need. We concentrate on 10 cutting-edge jailbreak strategies across three categories, 1525 questions from 61 specific harmful categories, and 13 popular LLMs. We adopt multi-dimensional metrics such as Attack Success Rate (ASR), Toxicity Score, Fluency, Token Length, and Grammatical Errors to thoroughly assess the LLMs' outputs under jailbreak. By normalizing and aggregating these metrics, we present a detailed reliability score for different LLMs, coupled with strategic recommendations to reduce their susceptibility to such vulnerabilities. Additionally, we explore the relationships among the models, attack strategies, and types of harmful content, as well as the correlations between the evaluation metrics, which proves the validity of our multifaceted evaluation framework. Our extensive experimental results demonstrate a lack of resilience among all tested LLMs against certain strategies, and highlight the need to concentrate on the reliability facets of LLMs. We believe our study can provide valuable insights into enhancing the security evaluation of LLMs against jailbreak within the domain.



## **44. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09093v1) [paper-pdf](http://arxiv.org/pdf/2408.09093v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **45. Can Editing LLMs Inject Harm?**

cs.CL

The first two authors contributed equally. 9 pages for main paper, 36  pages including appendix. The code, results, dataset for this paper and more  resources are on the project website: https://llm-editing.github.io

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.20224v3) [paper-pdf](http://arxiv.org/pdf/2407.20224v3)

**Authors**: Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, Kai Shu

**Abstract**: Knowledge editing has been increasingly adopted to correct the false or outdated knowledge in Large Language Models (LLMs). Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.



## **46. Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models**

cs.AI

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08989v1) [paper-pdf](http://arxiv.org/pdf/2408.08989v1)

**Authors**: Qingyuan Zeng, Zhenzhong Wang, Yiu-ming Cheung, Min Jiang

**Abstract**: While image-to-text models have demonstrated significant advancements in various vision-language tasks, they remain susceptible to adversarial attacks. Existing white-box attacks on image-to-text models require access to the architecture, gradients, and parameters of the target model, resulting in low practicality. Although the recently proposed gray-box attacks have improved practicality, they suffer from semantic loss during the training process, which limits their targeted attack performance. To advance adversarial attacks of image-to-text models, this paper focuses on a challenging scenario: decision-based black-box targeted attacks where the attackers only have access to the final output text and aim to perform targeted attacks. Specifically, we formulate the decision-based black-box targeted attack as a large-scale optimization problem. To efficiently solve the optimization problem, a three-stage process \textit{Ask, Attend, Attack}, called \textit{AAA}, is proposed to coordinate with the solver. \textit{Ask} guides attackers to create target texts that satisfy the specific semantics. \textit{Attend} identifies the crucial regions of the image for attacking, thus reducing the search space for the subsequent \textit{Attack}. \textit{Attack} uses an evolutionary algorithm to attack the crucial regions, where the attacks are semantically related to the target texts of \textit{Ask}, thus achieving targeted attacks without semantic loss. Experimental results on transformer-based and CNN+RNN-based image-to-text models confirmed the effectiveness of our proposed \textit{AAA}.



## **47. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08685v1) [paper-pdf](http://arxiv.org/pdf/2408.08685v1)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial perturbations, especially for topology attacks, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attack. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph.



## **48. MIA-Tuner: Adapting Large Language Models as Pre-training Text Detector**

cs.CL

code and dataset: https://github.com/wjfu99/MIA-Tuner

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08661v1) [paper-pdf](http://arxiv.org/pdf/2408.08661v1)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: The increasing parameters and expansive dataset of large language models (LLMs) highlight the urgent demand for a technical solution to audit the underlying privacy risks and copyright issues associated with LLMs. Existing studies have partially addressed this need through an exploration of the pre-training data detection problem, which is an instance of a membership inference attack (MIA). This problem involves determining whether a given piece of text has been used during the pre-training phase of the target LLM. Although existing methods have designed various sophisticated MIA score functions to achieve considerable detection performance in pre-trained LLMs, how to achieve high-confidence detection and how to perform MIA on aligned LLMs remain challenging. In this paper, we propose MIA-Tuner, a novel instruction-based MIA method, which instructs LLMs themselves to serve as a more precise pre-training data detector internally, rather than design an external MIA score function. Furthermore, we design two instruction-based safeguards to respectively mitigate the privacy risks brought by the existing methods and MIA-Tuner. To comprehensively evaluate the most recent state-of-the-art LLMs, we collect a more up-to-date MIA benchmark dataset, named WIKIMIA-24, to replace the widely adopted benchmark WIKIMIA. We conduct extensive experiments across various aligned and unaligned LLMs over the two benchmark datasets. The results demonstrate that MIA-Tuner increases the AUC of MIAs from 0.7 to a significantly high level of 0.9.



## **49. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

cs.IR

Survey paper

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.06992v2) [paper-pdf](http://arxiv.org/pdf/2407.06992v2)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.



## **50. ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages**

cs.CL

Accepted by ACL 2024 Main Conference

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2402.10753v2) [paper-pdf](http://arxiv.org/pdf/2402.10753v2)

**Authors**: Junjie Ye, Sixian Li, Guanyu Li, Caishuang Huang, Songyang Gao, Yilong Wu, Qi Zhang, Tao Gui, Xuanjing Huang

**Abstract**: Tool learning is widely acknowledged as a foundational approach or deploying large language models (LLMs) in real-world scenarios. While current research primarily emphasizes leveraging tools to augment LLMs, it frequently neglects emerging safety considerations tied to their application. To fill this gap, we present *ToolSword*, a comprehensive framework dedicated to meticulously investigating safety issues linked to LLMs in tool learning. Specifically, ToolSword delineates six safety scenarios for LLMs in tool learning, encompassing **malicious queries** and **jailbreak attacks** in the input stage, **noisy misdirection** and **risky cues** in the execution stage, and **harmful feedback** and **error conflicts** in the output stage. Experiments conducted on 11 open-source and closed-source LLMs reveal enduring safety challenges in tool learning, such as handling harmful queries, employing risky tools, and delivering detrimental feedback, which even GPT-4 is susceptible to. Moreover, we conduct further studies with the aim of fostering research on tool learning safety. The data is released in https://github.com/Junjie-Ye/ToolSword.



