# Latest Adversarial Attack Papers
**update at 2024-10-31 09:51:21**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2305.17000v7) [paper-pdf](http://arxiv.org/pdf/2305.17000v7)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **2. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23091v1) [paper-pdf](http://arxiv.org/pdf/2410.23091v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark).



## **3. Robustifying automatic speech recognition by extracting slowly varying features**

eess.AS

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2112.07400v2) [paper-pdf](http://arxiv.org/pdf/2112.07400v2)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: In the past few years, it has been shown that deep learning systems are highly vulnerable under attacks with adversarial examples. Neural-network-based automatic speech recognition (ASR) systems are no exception. Targeted and untargeted attacks can modify an audio input signal in such a way that humans still recognise the same words, while ASR systems are steered to predict a different transcription. In this paper, we propose a defense mechanism against targeted adversarial attacks consisting in removing fast-changing features from the audio signals, either by applying slow feature analysis, a low-pass filter, or both, before feeding the input to the ASR system. We perform an empirical analysis of hybrid ASR models trained on data pre-processed in such a way. While the resulting models perform quite well on benign data, they are significantly more robust against targeted adversarial attacks: Our final, proposed model shows a performance on clean data similar to the baseline model, while being more than four times more robust.



## **4. Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections**

cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.03052v2) [paper-pdf](http://arxiv.org/pdf/2406.03052v2)

**Authors**: Zihan Luo, Hong Huang, Yongkang Zhou, Jiping Zhang, Nuo Chen, Hai Jin

**Abstract**: Despite the remarkable capabilities demonstrated by Graph Neural Networks (GNNs) in graph-related tasks, recent research has revealed the fairness vulnerabilities in GNNs when facing malicious adversarial attacks. However, all existing fairness attacks require manipulating the connectivity between existing nodes, which may be prohibited in reality. To this end, we introduce a Node Injection-based Fairness Attack (NIFA), exploring the vulnerabilities of GNN fairness in such a more realistic setting. In detail, NIFA first designs two insightful principles for node injection operations, namely the uncertainty-maximization principle and homophily-increase principle, and then optimizes injected nodes' feature matrix to further ensure the effectiveness of fairness attacks. Comprehensive experiments on three real-world datasets consistently demonstrate that NIFA can significantly undermine the fairness of mainstream GNNs, even including fairness-aware GNNs, by injecting merely 1% of nodes. We sincerely hope that our work can stimulate increasing attention from researchers on the vulnerability of GNN fairness, and encourage the development of corresponding defense mechanisms. Our code and data are released at: https://github.com/CGCL-codes/NIFA.



## **5. Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector**

cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22888v1) [paper-pdf](http://arxiv.org/pdf/2410.22888v1)

**Authors**: Youcheng Huang, Fengbin Zhu, Jingkun Tang, Pan Zhou, Wenqiang Lei, Jiancheng Lv, Tat-Seng Chua

**Abstract**: Visual Language Models (VLMs) are vulnerable to adversarial attacks, especially those from adversarial images, which is however under-explored in literature. To facilitate research on this critical safety problem, we first construct a new laRge-scale Adervsarial images dataset with Diverse hArmful Responses (RADAR), given that existing datasets are either small-scale or only contain limited types of harmful responses. With the new RADAR dataset, we further develop a novel and effective iN-time Embedding-based AdveRSarial Image DEtection (NEARSIDE) method, which exploits a single vector that distilled from the hidden states of VLMs, which we call the attacking direction, to achieve the detection of adversarial images against benign ones in the input. Extensive experiments with two victim VLMs, LLaVA and MiniGPT-4, well demonstrate the effectiveness, efficiency, and cross-model transferrability of our proposed method. Our code is available at https://github.com/mob-scu/RADAR-NEARSIDE



## **6. Stealing User Prompts from Mixture of Experts**

cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22884v1) [paper-pdf](http://arxiv.org/pdf/2410.22884v1)

**Authors**: Itay Yona, Ilia Shumailov, Jamie Hayes, Nicholas Carlini

**Abstract**: Mixture-of-Experts (MoE) models improve the efficiency and scalability of dense language models by routing each token to a small number of experts in each layer. In this paper, we show how an adversary that can arrange for their queries to appear in the same batch of examples as a victim's queries can exploit Expert-Choice-Routing to fully disclose a victim's prompt. We successfully demonstrate the effectiveness of this attack on a two-layer Mixtral model, exploiting the tie-handling behavior of the torch.topk CUDA implementation. Our results show that we can extract the entire prompt using $O({VM}^2)$ queries (with vocabulary size $V$ and prompt length $M$) or 100 queries on average per token in the setting we consider. This is the first attack to exploit architectural flaws for the purpose of extracting user prompts, introducing a new class of LLM vulnerabilities.



## **7. Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation**

cs.IR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22844v1) [paper-pdf](http://arxiv.org/pdf/2410.22844v1)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Adversarial Collaborative Filtering (ACF), which typically applies adversarial perturbations at user and item embeddings through adversarial training, is widely recognized as an effective strategy for enhancing the robustness of Collaborative Filtering (CF) recommender systems against poisoning attacks. Besides, numerous studies have empirically shown that ACF can also improve recommendation performance compared to traditional CF. Despite these empirical successes, the theoretical understanding of ACF's effectiveness in terms of both performance and robustness remains unclear. To bridge this gap, in this paper, we first theoretically show that ACF can achieve a lower recommendation error compared to traditional CF with the same training epochs in both clean and poisoned data contexts. Furthermore, by establishing bounds for reductions in recommendation error during ACF's optimization process, we find that applying personalized magnitudes of perturbation for different users based on their embedding scales can further improve ACF's effectiveness. Building on these theoretical understandings, we propose Personalized Magnitude Adversarial Collaborative Filtering (PamaCF). Extensive experiments demonstrate that PamaCF effectively defends against various types of poisoning attacks while significantly enhancing recommendation performance.



## **8. One Prompt to Verify Your Models: Black-Box Text-to-Image Models Verification via Non-Transferable Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22725v1) [paper-pdf](http://arxiv.org/pdf/2410.22725v1)

**Authors**: Ji Guo, Wenbo Jiang, Rui Zhang, Guoming Lu, Hongwei Li, Weiren Wu

**Abstract**: Recently, the success of Text-to-Image (T2I) models has led to the rise of numerous third-party platforms, which claim to provide cheaper API services and more flexibility in model options. However, this also raises a new security concern: Are these third-party services truly offering the models they claim? To address this problem, we propose the first T2I model verification method named Text-to-Image Model Verification via Non-Transferable Adversarial Attacks (TVN). The non-transferability of adversarial examples means that these examples are only effective on a target model and ineffective on other models, thereby allowing for the verification of the target model. TVN utilizes the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to optimize the cosine similarity of a prompt's text encoding, generating non-transferable adversarial prompts. By calculating the CLIP-text scores between the non-transferable adversarial prompts without perturbations and the images, we can verify if the model matches the claimed target model, based on a 3-sigma threshold. The experiments showed that TVN performed well in both closed-set and open-set scenarios, achieving a verification accuracy of over 90\%. Moreover, the adversarial prompts generated by TVN significantly reduced the CLIP-text scores of the target model, while having little effect on other models.



## **9. Geometry Cloak: Preventing TGS-based 3D Reconstruction from Copyrighted Images**

cs.CV

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22705v1) [paper-pdf](http://arxiv.org/pdf/2410.22705v1)

**Authors**: Qi Song, Ziyuan Luo, Ka Chun Cheung, Simon See, Renjie Wan

**Abstract**: Single-view 3D reconstruction methods like Triplane Gaussian Splatting (TGS) have enabled high-quality 3D model generation from just a single image input within seconds. However, this capability raises concerns about potential misuse, where malicious users could exploit TGS to create unauthorized 3D models from copyrighted images. To prevent such infringement, we propose a novel image protection approach that embeds invisible geometry perturbations, termed "geometry cloaks", into images before supplying them to TGS. These carefully crafted perturbations encode a customized message that is revealed when TGS attempts 3D reconstructions of the cloaked image. Unlike conventional adversarial attacks that simply degrade output quality, our method forces TGS to fail the 3D reconstruction in a specific way - by generating an identifiable customized pattern that acts as a watermark. This watermark allows copyright holders to assert ownership over any attempted 3D reconstructions made from their protected images. Extensive experiments have verified the effectiveness of our geometry cloak. Our project is available at https://qsong2001.github.io/geometry_cloak.



## **10. Backdoor Attack Against Vision Transformers via Attention Gradient-Based Image Erosion**

cs.CV

Accepted by IEEE GLOBECOM 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22678v1) [paper-pdf](http://arxiv.org/pdf/2410.22678v1)

**Authors**: Ji Guo, Hongwei Li, Wenbo Jiang, Guoming Lu

**Abstract**: Vision Transformers (ViTs) have outperformed traditional Convolutional Neural Networks (CNN) across various computer vision tasks. However, akin to CNN, ViTs are vulnerable to backdoor attacks, where the adversary embeds the backdoor into the victim model, causing it to make wrong predictions about testing samples containing a specific trigger. Existing backdoor attacks against ViTs have the limitation of failing to strike an optimal balance between attack stealthiness and attack effectiveness.   In this work, we propose an Attention Gradient-based Erosion Backdoor (AGEB) targeted at ViTs. Considering the attention mechanism of ViTs, AGEB selectively erodes pixels in areas of maximal attention gradient, embedding a covert backdoor trigger. Unlike previous backdoor attacks against ViTs, AGEB achieves an optimal balance between attack stealthiness and attack effectiveness, ensuring the trigger remains invisible to human detection while preserving the model's accuracy on clean samples. Extensive experimental evaluations across various ViT architectures and datasets confirm the effectiveness of AGEB, achieving a remarkable Attack Success Rate (ASR) without diminishing Clean Data Accuracy (CDA). Furthermore, the stealthiness of AGEB is rigorously validated, demonstrating minimal visual discrepancies between the clean and the triggered images.



## **11. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

cs.SE

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22663v1) [paper-pdf](http://arxiv.org/pdf/2410.22663v1)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains, such as toxicity detection, chatbot consulting, and review analysis. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Several studies indicate that traditional metrics, such as model confidence and accuracy, are insufficient to build human trust in ML models. These models often learn spurious correlations during training and predict based on them during inference. In the real world, where such correlations are absent, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods, which is time-consuming and not scalable. We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers, which automatically checks whether the prediction-contributing words are related to the predicted class using explanation methods and word embeddings. To demonstrate its practical usefulness, we introduce a novel adversarial attack method targeting trustworthiness issues identified by TOKI. We compare TOKI with a naive baseline based solely on model confidence using human-created ground truths of 6,000 predictions. We also compare TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided adversarial attack method is more effective with fewer perturbations than A2T.



## **12. AdvWeb: Controllable Black-box Attacks on VLM-powered Web Agents**

cs.CR

15 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.17401v2) [paper-pdf](http://arxiv.org/pdf/2410.17401v2)

**Authors**: Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, Bo Li

**Abstract**: Vision Language Models (VLMs) have revolutionized the creation of generalist web agents, empowering them to autonomously complete diverse tasks on real-world websites, thereby boosting human efficiency and productivity. However, despite their remarkable capabilities, the safety and security of these agents against malicious attacks remain critically underexplored, raising significant concerns about their safe deployment. To uncover and exploit such vulnerabilities in web agents, we provide AdvWeb, a novel black-box attack framework designed against web agents. AdvWeb trains an adversarial prompter model that generates and injects adversarial prompts into web pages, misleading web agents into executing targeted adversarial actions such as inappropriate stock purchases or incorrect bank transactions, actions that could lead to severe real-world consequences. With only black-box access to the web agent, we train and optimize the adversarial prompter model using DPO, leveraging both successful and failed attack strings against the target agent. Unlike prior approaches, our adversarial string injection maintains stealth and control: (1) the appearance of the website remains unchanged before and after the attack, making it nearly impossible for users to detect tampering, and (2) attackers can modify specific substrings within the generated adversarial string to seamlessly change the attack objective (e.g., purchasing stocks from a different company), enhancing attack flexibility and efficiency. We conduct extensive evaluations, demonstrating that AdvWeb achieves high success rates in attacking SOTA GPT-4V-based VLM agent across various web tasks. Our findings expose critical vulnerabilities in current LLM/VLM-based agents, emphasizing the urgent need for developing more reliable web agents and effective defenses. Our code and data are available at https://ai-secure.github.io/AdvWeb/ .



## **13. LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate**

cs.CV

NeurIPS 2024 Camera Ready

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2405.13985v2) [paper-pdf](http://arxiv.org/pdf/2405.13985v2)

**Authors**: Anthony Fuller, Daniel G. Kyrollos, Yousef Yassin, James R. Green

**Abstract**: High-resolution images offer more information about scenes that can improve model accuracy. However, the dominant model architecture in computer vision, the vision transformer (ViT), cannot effectively leverage larger images without finetuning -- ViTs poorly extrapolate to more patches at test time, although transformers offer sequence length flexibility. We attribute this shortcoming to the current patch position encoding methods, which create a distribution shift when extrapolating.   We propose a drop-in replacement for the position encoding of plain ViTs that restricts attention heads to fixed fields of view, pointed in different directions, using 2D attention masks. Our novel method, called LookHere, provides translation-equivariance, ensures attention head diversity, and limits the distribution shift that attention heads face when extrapolating. We demonstrate that LookHere improves performance on classification (avg. 1.6%), against adversarial attack (avg. 5.4%), and decreases calibration error (avg. 1.5%) -- on ImageNet without extrapolation. With extrapolation, LookHere outperforms the current SoTA position encoding method, 2D-RoPE, by 21.7% on ImageNet when trained at $224^2$ px and tested at $1024^2$ px. Additionally, we release a high-resolution test set to improve the evaluation of high-resolution image classifiers, called ImageNet-HR.



## **14. Power side-channel leakage localization through adversarial training of deep neural networks**

cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22425v1) [paper-pdf](http://arxiv.org/pdf/2410.22425v1)

**Authors**: Jimmy Gammell, Anand Raghunathan, Kaushik Roy

**Abstract**: Supervised deep learning has emerged as an effective tool for carrying out power side-channel attacks on cryptographic implementations. While increasingly-powerful deep learning-based attacks are regularly published, comparatively-little work has gone into using deep learning to defend against these attacks. In this work we propose a technique for identifying which timesteps in a power trace are responsible for leaking a cryptographic key, through an adversarial game between a deep learning-based side-channel attacker which seeks to classify a sensitive variable from the power traces recorded during encryption, and a trainable noise generator which seeks to thwart this attack by introducing a minimal amount of noise into the power traces. We demonstrate on synthetic datasets that our method can outperform existing techniques in the presence of common countermeasures such as Boolean masking and trace desynchronization. Results on real datasets are weak because the technique is highly sensitive to hyperparameters and early-stop point, and we lack a holdout dataset with ground truth knowledge of leaking points for model selection. Nonetheless, we believe our work represents an important first step towards deep side-channel leakage localization without relying on strong assumptions about the implementation or the nature of its leakage. An open-source PyTorch implementation of our experiments is provided.



## **15. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

cs.LG

20 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22307v1) [paper-pdf](http://arxiv.org/pdf/2410.22307v1)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: Open-source Large Language Models (LLMs) have recently demonstrated remarkable capabilities in natural language understanding and generation, leading to widespread adoption across various domains. However, their increasing model sizes render local deployment impractical for individual users, pushing many to rely on computing service providers for inference through a blackbox API. This reliance introduces a new risk: a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby delivering inferior outputs while benefiting from cost savings. In this paper, we formalize the problem of verifiable inference for LLMs. Existing verifiable computing solutions based on cryptographic or game-theoretic techniques are either computationally uneconomical or rest on strong assumptions. We introduce SVIP, a secret-based verifiable LLM inference protocol that leverages intermediate outputs from LLM as unique model identifiers. By training a proxy task on these outputs and requiring the computing provider to return both the generated text and the processed intermediate outputs, users can reliably verify whether the computing provider is acting honestly. In addition, the integration of a secret mechanism further enhances the security of our protocol. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per query for verification.



## **16. Embedding-based classifiers can detect prompt injection attacks**

cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22284v1) [paper-pdf](http://arxiv.org/pdf/2410.22284v1)

**Authors**: Md. Ahsan Ayub, Subhabrata Majumdar

**Abstract**: Large Language Models (LLMs) are seeing significant adoption in every type of organization due to their exceptional generative capabilities. However, LLMs are found to be vulnerable to various adversarial attacks, particularly prompt injection attacks, which trick them into producing harmful or inappropriate content. Adversaries execute such attacks by crafting malicious prompts to deceive the LLMs. In this paper, we propose a novel approach based on embedding-based Machine Learning (ML) classifiers to protect LLM-based applications against this severe threat. We leverage three commonly used embedding models to generate embeddings of malicious and benign prompts and utilize ML classifiers to predict whether an input prompt is malicious. Out of several traditional ML methods, we achieve the best performance with classifiers built using Random Forest and XGBoost. Our classifiers outperform state-of-the-art prompt injection classifiers available in open-source implementations, which use encoder-only neural networks.



## **17. AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts**

cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22143v1) [paper-pdf](http://arxiv.org/pdf/2410.22143v1)

**Authors**: Vishal Kumar, Zeyi Liao, Jaylen Jones, Huan Sun

**Abstract**: Although large language models (LLMs) are typically aligned, they remain vulnerable to jailbreaking through either carefully crafted prompts in natural language or, interestingly, gibberish adversarial suffixes. However, gibberish tokens have received relatively less attention despite their success in attacking aligned LLMs. Recent work, AmpleGCG~\citep{liao2024amplegcg}, demonstrates that a generative model can quickly produce numerous customizable gibberish adversarial suffixes for any harmful query, exposing a range of alignment gaps in out-of-distribution (OOD) language spaces. To bring more attention to this area, we introduce AmpleGCG-Plus, an enhanced version that achieves better performance in fewer attempts. Through a series of exploratory experiments, we identify several training strategies to improve the learning of gibberish suffixes. Our results, verified under a strict evaluation setting, show that it outperforms AmpleGCG on both open-weight and closed-source models, achieving increases in attack success rate (ASR) of up to 17\% in the white-box setting against Llama-2-7B-chat, and more than tripling ASR in the black-box setting against GPT-4. Notably, AmpleGCG-Plus jailbreaks the newer GPT-4o series of models at similar rates to GPT-4, and, uncovers vulnerabilities against the recently proposed circuit breakers defense. We publicly release AmpleGCG-Plus along with our collected training datasets.



## **18. Iterative Window Mean Filter: Thwarting Diffusion-based Adversarial Purification**

cs.CR

Accepted in IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2408.10673v3) [paper-pdf](http://arxiv.org/pdf/2408.10673v3)

**Authors**: Hanrui Wang, Ruoxi Sun, Cunjian Chen, Minhui Xue, Lay-Ki Soon, Shuo Wang, Zhe Jin

**Abstract**: Face authentication systems have brought significant convenience and advanced developments, yet they have become unreliable due to their sensitivity to inconspicuous perturbations, such as adversarial attacks. Existing defenses often exhibit weaknesses when facing various attack algorithms and adaptive attacks or compromise accuracy for enhanced security. To address these challenges, we have developed a novel and highly efficient non-deep-learning-based image filter called the Iterative Window Mean Filter (IWMF) and proposed a new framework for adversarial purification, named IWMF-Diff, which integrates IWMF and denoising diffusion models. These methods can function as pre-processing modules to eliminate adversarial perturbations without necessitating further modifications or retraining of the target system. We demonstrate that our proposed methodologies fulfill four critical requirements: preserved accuracy, improved security, generalizability to various threats in different settings, and better resistance to adaptive attacks. This performance surpasses that of the state-of-the-art adversarial purification method, DiffPure.



## **19. Forging the Forger: An Attempt to Improve Authorship Verification via Data Augmentation**

cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2403.11265v2) [paper-pdf](http://arxiv.org/pdf/2403.11265v2)

**Authors**: Silvia Corbara, Alejandro Moreo

**Abstract**: Authorship Verification (AV) is a text classification task concerned with inferring whether a candidate text has been written by one specific author or by someone else. It has been shown that many AV systems are vulnerable to adversarial attacks, where a malicious author actively tries to fool the classifier by either concealing their writing style, or by imitating the style of another author. In this paper, we investigate the potential benefits of augmenting the classifier training set with (negative) synthetic examples. These synthetic examples are generated to imitate the style of the author of interest. We analyze the improvements in classifier prediction that this augmentation brings to bear in the task of AV in an adversarial setting. In particular, we experiment with three different generator architectures (one based on Recurrent Neural Networks, another based on small-scale transformers, and another based on the popular GPT model) and with two training strategies (one inspired by standard Language Models, and another inspired by Wasserstein Generative Adversarial Networks). We evaluate our hypothesis on five datasets (three of which have been specifically collected to represent an adversarial setting) and using two learning algorithms for the AV classifier (Support Vector Machines and Convolutional Neural Networks). This experimentation has yielded negative results, revealing that, although our methodology proves effective in many adversarial settings, its benefits are too sporadic for a pragmatical application.



## **20. On the Robustness of Adversarial Training Against Uncertainty Attacks**

cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21952v1) [paper-pdf](http://arxiv.org/pdf/2410.21952v1)

**Authors**: Emanuele Ledda, Giovanni Scodeller, Daniele Angioni, Giorgio Piras, Antonio Emanuele Cinà, Giorgio Fumera, Battista Biggio, Fabio Roli

**Abstract**: In learning problems, the noise inherent to the task at hand hinders the possibility to infer without a certain degree of uncertainty. Quantifying this uncertainty, regardless of its wide use, assumes high relevance for security-sensitive applications. Within these scenarios, it becomes fundamental to guarantee good (i.e., trustworthy) uncertainty measures, which downstream modules can securely employ to drive the final decision-making process. However, an attacker may be interested in forcing the system to produce either (i) highly uncertain outputs jeopardizing the system's availability or (ii) low uncertainty estimates, making the system accept uncertain samples that would instead require a careful inspection (e.g., human intervention). Therefore, it becomes fundamental to understand how to obtain robust uncertainty estimates against these kinds of attacks. In this work, we reveal both empirically and theoretically that defending against adversarial examples, i.e., carefully perturbed samples that cause misclassification, additionally guarantees a more secure, trustworthy uncertainty estimate under common attack scenarios without the need for an ad-hoc defense strategy. To support our claims, we evaluate multiple adversarial-robust models from the publicly available benchmark RobustBench on the CIFAR-10 and ImageNet datasets.



## **21. Enhancing Adversarial Attacks through Chain of Thought**

cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21791v1) [paper-pdf](http://arxiv.org/pdf/2410.21791v1)

**Authors**: Jingbo Su

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across various domains but remain susceptible to safety concerns. Prior research indicates that gradient-based adversarial attacks are particularly effective against aligned LLMs and the chain of thought (CoT) prompting can elicit desired answers through step-by-step reasoning. This paper proposes enhancing the robustness of adversarial attacks on aligned LLMs by integrating CoT prompts with the greedy coordinate gradient (GCG) technique. Using CoT triggers instead of affirmative targets stimulates the reasoning abilities of backend LLMs, thereby improving the transferability and universality of adversarial attacks. We conducted an ablation study comparing our CoT-GCG approach with Amazon Web Services auto-cot. Results revealed our approach outperformed both the baseline GCG attack and CoT prompting. Additionally, we used Llama Guard to evaluate potentially harmful interactions, providing a more objective risk assessment of entire conversations compared to matching outputs to rejection phrases. The code of this paper is available at https://github.com/sujingbo0217/CS222W24-LLM-Attack.



## **22. Transferable Adversarial Attacks on SAM and Its Downstream Models**

cs.LG

This work is accepted by Neurips2024

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.20197v2) [paper-pdf](http://arxiv.org/pdf/2410.20197v2)

**Authors**: Song Xia, Wenhan Yang, Yi Yu, Xun Lin, Henghui Ding, Lingyu Duan, Xudong Jiang

**Abstract**: The utilization of large foundational models has a dilemma: while fine-tuning downstream tasks from them holds promise for making use of the well-generalized knowledge in practical applications, their open accessibility also poses threats of adverse usage. This paper, for the first time, explores the feasibility of adversarial attacking various downstream models fine-tuned from the segment anything model (SAM), by solely utilizing the information from the open-sourced SAM. In contrast to prevailing transfer-based adversarial attacks, we demonstrate the existence of adversarial dangers even without accessing the downstream task and dataset to train a similar surrogate model. To enhance the effectiveness of the adversarial attack towards models fine-tuned on unknown datasets, we propose a universal meta-initialization (UMI) algorithm to extract the intrinsic vulnerability inherent in the foundation model, which is then utilized as the prior knowledge to guide the generation of adversarial perturbations. Moreover, by formulating the gradient difference in the attacking process between the open-sourced SAM and its fine-tuned downstream models, we theoretically demonstrate that a deviation occurs in the adversarial update direction by directly maximizing the distance of encoded feature embeddings in the open-sourced SAM. Consequently, we propose a gradient robust loss that simulates the associated uncertainty with gradient-based noise augmentation to enhance the robustness of generated adversarial examples (AEs) towards this deviation, thus improving the transferability. Extensive experiments demonstrate the effectiveness of the proposed universal meta-initialized and gradient robust adversarial attack (UMI-GRAT) toward SAMs and their downstream models. Code is available at https://github.com/xiasong0501/GRAT.



## **23. CFSafety: Comprehensive Fine-grained Safety Assessment for LLMs**

cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21695v1) [paper-pdf](http://arxiv.org/pdf/2410.21695v1)

**Authors**: Zhihao Liu, Chenhui Hu

**Abstract**: As large language models (LLMs) rapidly evolve, they bring significant conveniences to our work and daily lives, but also introduce considerable safety risks. These models can generate texts with social biases or unethical content, and under specific adversarial instructions, may even incite illegal activities. Therefore, rigorous safety assessments of LLMs are crucial. In this work, we introduce a safety assessment benchmark, CFSafety, which integrates 5 classic safety scenarios and 5 types of instruction attacks, totaling 10 categories of safety questions, to form a test set with 25k prompts. This test set was used to evaluate the natural language generation (NLG) capabilities of LLMs, employing a combination of simple moral judgment and a 1-5 safety rating scale for scoring. Using this benchmark, we tested eight popular LLMs, including the GPT series. The results indicate that while GPT-4 demonstrated superior safety performance, the safety effectiveness of LLMs, including this model, still requires improvement. The data and code associated with this study are available on GitHub.



## **24. AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models**

cs.CV

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21471v1) [paper-pdf](http://arxiv.org/pdf/2410.21471v1)

**Authors**: Yaopei Zeng, Yuanpu Cao, Bochuan Cao, Yurui Chang, Jinghui Chen, Lu Lin

**Abstract**: Recent advances in diffusion models have significantly enhanced the quality of image synthesis, yet they have also introduced serious safety concerns, particularly the generation of Not Safe for Work (NSFW) content. Previous research has demonstrated that adversarial prompts can be used to generate NSFW content. However, such adversarial text prompts are often easily detectable by text-based filters, limiting their efficacy. In this paper, we expose a previously overlooked vulnerability: adversarial image attacks targeting Image-to-Image (I2I) diffusion models. We propose AdvI2I, a novel framework that manipulates input images to induce diffusion models to generate NSFW content. By optimizing a generator to craft adversarial images, AdvI2I circumvents existing defense mechanisms, such as Safe Latent Diffusion (SLD), without altering the text prompts. Furthermore, we introduce AdvI2I-Adaptive, an enhanced version that adapts to potential countermeasures and minimizes the resemblance between adversarial images and NSFW concept embeddings, making the attack more resilient against defenses. Through extensive experiments, we demonstrate that both AdvI2I and AdvI2I-Adaptive can effectively bypass current safeguards, highlighting the urgent need for stronger security measures to address the misuse of I2I diffusion models.



## **25. TACO: Adversarial Camouflage Optimization on Trucks to Fool Object Detectors**

cs.CV

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21443v1) [paper-pdf](http://arxiv.org/pdf/2410.21443v1)

**Authors**: Adonisz Dimitriu, Tamás Michaletzky, Viktor Remeli

**Abstract**: Adversarial attacks threaten the reliability of machine learning models in critical applications like autonomous vehicles and defense systems. As object detectors become more robust with models like YOLOv8, developing effective adversarial methodologies is increasingly challenging. We present Truck Adversarial Camouflage Optimization (TACO), a novel framework that generates adversarial camouflage patterns on 3D vehicle models to deceive state-of-the-art object detectors. Adopting Unreal Engine 5, TACO integrates differentiable rendering with a Photorealistic Rendering Network to optimize adversarial textures targeted at YOLOv8. To ensure the generated textures are both effective in deceiving detectors and visually plausible, we introduce the Convolutional Smooth Loss function, a generalized smooth loss function. Experimental evaluations demonstrate that TACO significantly degrades YOLOv8's detection performance, achieving an AP@0.5 of 0.0099 on unseen test data. Furthermore, these adversarial patterns exhibit strong transferability to other object detection models such as Faster R-CNN and earlier YOLO versions.



## **26. Securing Multi-turn Conversational Language Models From Distributed Backdoor Triggers**

cs.CL

Findings of EMNLP 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2407.04151v2) [paper-pdf](http://arxiv.org/pdf/2407.04151v2)

**Authors**: Terry Tong, Jiashu Xu, Qin Liu, Muhao Chen

**Abstract**: Large language models (LLMs) have acquired the ability to handle longer context lengths and understand nuances in text, expanding their dialogue capabilities beyond a single utterance. A popular user-facing application of LLMs is the multi-turn chat setting. Though longer chat memory and better understanding may seemingly benefit users, our paper exposes a vulnerability that leverages the multi-turn feature and strong learning ability of LLMs to harm the end-user: the backdoor. We demonstrate that LLMs can capture the combinational backdoor representation. Only upon presentation of triggers together does the backdoor activate. We also verify empirically that this representation is invariant to the position of the trigger utterance. Subsequently, inserting a single extra token into two utterances of 5%of the data can cause over 99% Attack Success Rate (ASR). Our results with 3 triggers demonstrate that this framework is generalizable, compatible with any trigger in an adversary's toolbox in a plug-and-play manner. Defending the backdoor can be challenging in the chat setting because of the large input and output space. Our analysis indicates that the distributed backdoor exacerbates the current challenges by polynomially increasing the dimension of the attacked input space. Canonical textual defenses like ONION and BKI leverage auxiliary model forward passes over individual tokens, scaling exponentially with the input sequence length and struggling to maintain computational feasibility. To this end, we propose a decoding time defense - decayed contrastive decoding - that scales linearly with assistant response sequence length and reduces the backdoor to as low as 0.35%.



## **27. Resilience in Knowledge Graph Embeddings**

cs.LG

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21163v1) [paper-pdf](http://arxiv.org/pdf/2410.21163v1)

**Authors**: Arnab Sharma, N'Dah Jean Kouagou, Axel-Cyrille Ngonga Ngomo

**Abstract**: In recent years, knowledge graphs have gained interest and witnessed widespread applications in various domains, such as information retrieval, question-answering, recommendation systems, amongst others. Large-scale knowledge graphs to this end have demonstrated their utility in effectively representing structured knowledge. To further facilitate the application of machine learning techniques, knowledge graph embedding (KGE) models have been developed. Such models can transform entities and relationships within knowledge graphs into vectors. However, these embedding models often face challenges related to noise, missing information, distribution shift, adversarial attacks, etc. This can lead to sub-optimal embeddings and incorrect inferences, thereby negatively impacting downstream applications. While the existing literature has focused so far on adversarial attacks on KGE models, the challenges related to the other critical aspects remain unexplored. In this paper, we, first of all, give a unified definition of resilience, encompassing several factors such as generalisation, performance consistency, distribution adaption, and robustness. After formalizing these concepts for machine learning in general, we define them in the context of knowledge graphs. To find the gap in the existing works on resilience in the context of knowledge graphs, we perform a systematic survey, taking into account all these aspects mentioned previously. Our survey results show that most of the existing works focus on a specific aspect of resilience, namely robustness. After categorizing such works based on their respective aspects of resilience, we discuss the challenges and future research directions.



## **28. Adversarial robustness of VAEs through the lens of local geometry**

cs.LG

International Conference on Artificial Intelligence and Statistics  (AISTATS) 2023

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2208.03923v3) [paper-pdf](http://arxiv.org/pdf/2208.03923v3)

**Authors**: Asif Khan, Amos Storkey

**Abstract**: In an unsupervised attack on variational autoencoders (VAEs), an adversary finds a small perturbation in an input sample that significantly changes its latent space encoding, thereby compromising the reconstruction for a fixed decoder. A known reason for such vulnerability is the distortions in the latent space resulting from a mismatch between approximated latent posterior and a prior distribution. Consequently, a slight change in an input sample can move its encoding to a low/zero density region in the latent space resulting in an unconstrained generation. This paper demonstrates that an optimal way for an adversary to attack VAEs is to exploit a directional bias of a stochastic pullback metric tensor induced by the encoder and decoder networks. The pullback metric tensor of an encoder measures the change in infinitesimal latent volume from an input to a latent space. Thus, it can be viewed as a lens to analyse the effect of input perturbations leading to latent space distortions. We propose robustness evaluation scores using the eigenspectrum of a pullback metric tensor. Moreover, we empirically show that the scores correlate with the robustness parameter $\beta$ of the $\beta-$VAE. Since increasing $\beta$ also degrades reconstruction quality, we demonstrate a simple alternative using \textit{mixup} training to fill the empty regions in the latent space, thus improving robustness with improved reconstruction.



## **29. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models**

cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20940v1) [paper-pdf](http://arxiv.org/pdf/2410.20940v1)

**Authors**: Piotr Przybyła

**Abstract**: We investigate the challenge of generating adversarial examples to test the robustness of text classification algorithms detecting low-credibility content, including propaganda, false claims, rumours and hyperpartisan news. We focus on simulation of content moderation by setting realistic limits on the number of queries an attacker is allowed to attempt. Within our solution (TREPAT), initial rephrasings are generated by large language models with prompts inspired by meaning-preserving NLP tasks, e.g. text simplification and style transfer. Subsequently, these modifications are decomposed into small changes, applied through beam search procedure until the victim classifier changes its decision. The evaluation confirms the superiority of our approach in the constrained scenario, especially in case of long input text (news articles), where exhaustive search is not feasible.



## **30. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

cs.CR

v0.1

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20911v1) [paper-pdf](http://arxiv.org/pdf/2410.20911v1)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis



## **31. Evaluating the Robustness of LiDAR Point Cloud Tracking Against Adversarial Attack**

cs.CV

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20893v1) [paper-pdf](http://arxiv.org/pdf/2410.20893v1)

**Authors**: Shengjing Tian, Yinan Han, Xiantong Zhao, Bin Liu, Xiuping Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.



## **32. GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models**

cs.LG

10 pages, 13 figures

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2304.09875v3) [paper-pdf](http://arxiv.org/pdf/2304.09875v3)

**Authors**: Zaitang Li, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Current studies on adversarial robustness mainly focus on aggregating local robustness results from a set of data samples to evaluate and rank different models. However, the local statistics may not well represent the true global robustness of the underlying unknown data distribution. To address this challenge, this paper makes the first attempt to present a new framework, called GREAT Score , for global robustness evaluation of adversarial perturbation using generative models. Formally, GREAT Score carries the physical meaning of a global statistic capturing a mean certified attack-proof perturbation level over all samples drawn from a generative model. For finite-sample evaluation, we also derive a probabilistic guarantee on the sample complexity and the difference between the sample mean and the true mean. GREAT Score has several advantages: (1) Robustness evaluations using GREAT Score are efficient and scalable to large models, by sparing the need of running adversarial attacks. In particular, we show high correlation and significantly reduced computation cost of GREAT Score when compared to the attack-based model ranking on RobustBench (Croce,et. al. 2021). (2) The use of generative models facilitates the approximation of the unknown data distribution. In our ablation study with different generative adversarial networks (GANs), we observe consistency between global robustness evaluation and the quality of GANs. (3) GREAT Score can be used for remote auditing of privacy-sensitive black-box models, as demonstrated by our robustness evaluation on several online facial recognition services.



## **33. Robust Text Classification: Analyzing Prototype-Based Networks**

cs.CL

Published at EMNLP Findings 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2311.06647v3) [paper-pdf](http://arxiv.org/pdf/2311.06647v3)

**Authors**: Zhivar Sourati, Darshan Deshpande, Filip Ilievski, Kiril Gashteovski, Sascha Saralajew

**Abstract**: Downstream applications often require text classification models to be accurate and robust. While the accuracy of the state-of-the-art Language Models (LMs) approximates human performance, they often exhibit a drop in performance on noisy data found in the real world. This lack of robustness can be concerning, as even small perturbations in the text, irrelevant to the target task, can cause classifiers to incorrectly change their predictions. A potential solution can be the family of Prototype-Based Networks (PBNs) that classifies examples based on their similarity to prototypical examples of a class (prototypes) and has been shown to be robust to noise for computer vision tasks. In this paper, we study whether the robustness properties of PBNs transfer to text classification tasks under both targeted and static adversarial attack settings. Our results show that PBNs, as a mere architectural variation of vanilla LMs, offer more robustness compared to vanilla LMs under both targeted and static settings. We showcase how PBNs' interpretability can help us to understand PBNs' robustness properties. Finally, our ablation studies reveal the sensitivity of PBNs' robustness to how strictly clustering is done in the training phase, as tighter clustering results in less robust PBNs.



## **34. Meta-Learning Approaches for Improving Detection of Unseen Speech Deepfakes**

eess.AS

6 pages, accepted to the IEEE Spoken Language Technology Workshop  (SLT) 2024

**SubmitDate**: 2024-10-27    [abs](http://arxiv.org/abs/2410.20578v1) [paper-pdf](http://arxiv.org/pdf/2410.20578v1)

**Authors**: Ivan Kukanov, Janne Laakkonen, Tomi Kinnunen, Ville Hautamäki

**Abstract**: Current speech deepfake detection approaches perform satisfactorily against known adversaries; however, generalization to unseen attacks remains an open challenge. The proliferation of speech deepfakes on social media underscores the need for systems that can generalize to unseen attacks not observed during training. We address this problem from the perspective of meta-learning, aiming to learn attack-invariant features to adapt to unseen attacks with very few samples available. This approach is promising since generating of a high-scale training dataset is often expensive or infeasible. Our experiments demonstrated an improvement in the Equal Error Rate (EER) from 21.67% to 10.42% on the InTheWild dataset, using just 96 samples from the unseen dataset. Continuous few-shot adaptation ensures that the system remains up-to-date.



## **35. LLM Robustness Against Misinformation in Biomedical Question Answering**

cs.CL

**SubmitDate**: 2024-10-27    [abs](http://arxiv.org/abs/2410.21330v1) [paper-pdf](http://arxiv.org/pdf/2410.21330v1)

**Authors**: Alexander Bondarenko, Adrian Viehweger

**Abstract**: The retrieval-augmented generation (RAG) approach is used to reduce the confabulation of large language models (LLMs) for question answering by retrieving and providing additional context coming from external knowledge sources (e.g., by adding the context to the prompt). However, injecting incorrect information can mislead the LLM to generate an incorrect answer.   In this paper, we evaluate the effectiveness and robustness of four LLMs against misinformation - Gemma 2, GPT-4o-mini, Llama~3.1, and Mixtral - in answering biomedical questions. We assess the answer accuracy on yes-no and free-form questions in three scenarios: vanilla LLM answers (no context is provided), "perfect" augmented generation (correct context is provided), and prompt-injection attacks (incorrect context is provided). Our results show that Llama 3.1 (70B parameters) achieves the highest accuracy in both vanilla (0.651) and "perfect" RAG (0.802) scenarios. However, the accuracy gap between the models almost disappears with "perfect" RAG, suggesting its potential to mitigate the LLM's size-related effectiveness differences.   We further evaluate the ability of the LLMs to generate malicious context on one hand and the LLM's robustness against prompt-injection attacks on the other hand, using metrics such as attack success rate (ASR), accuracy under attack, and accuracy drop. As adversaries, we use the same four LLMs (Gemma 2, GPT-4o-mini, Llama 3.1, and Mixtral) to generate incorrect context that is injected in the target model's prompt. Interestingly, Llama is shown to be the most effective adversary, causing accuracy drops of up to 0.48 for vanilla answers and 0.63 for "perfect" RAG across target models. Our analysis reveals that robustness rankings vary depending on the evaluation measure, highlighting the complexity of assessing LLM resilience to adversarial attacks.



## **36. Integrating uncertainty quantification into randomized smoothing based robustness guarantees**

cs.LG

**SubmitDate**: 2024-10-27    [abs](http://arxiv.org/abs/2410.20432v1) [paper-pdf](http://arxiv.org/pdf/2410.20432v1)

**Authors**: Sina Däubener, Kira Maag, David Krueger, Asja Fischer

**Abstract**: Deep neural networks have proven to be extremely powerful, however, they are also vulnerable to adversarial attacks which can cause hazardous incorrect predictions in safety-critical applications. Certified robustness via randomized smoothing gives a probabilistic guarantee that the smoothed classifier's predictions will not change within an $\ell_2$-ball around a given input. On the other hand (uncertainty) score-based rejection is a technique often applied in practice to defend models against adversarial attacks. In this work, we fuse these two approaches by integrating a classifier that abstains from predicting when uncertainty is high into the certified robustness framework. This allows us to derive two novel robustness guarantees for uncertainty aware classifiers, namely (i) the radius of an $\ell_2$-ball around the input in which the same label is predicted and uncertainty remains low and (ii) the $\ell_2$-radius of a ball in which the predictions will either not change or be uncertain. While the former provides robustness guarantees with respect to attacks aiming at increased uncertainty, the latter informs about the amount of input perturbation necessary to lead the uncertainty aware model into a wrong prediction. Notably, this is on CIFAR10 up to 20.93% larger than for models not allowing for uncertainty based rejection. We demonstrate, that the novel framework allows for a systematic robustness evaluation of different network architectures and uncertainty measures and to identify desired properties of uncertainty quantification techniques. Moreover, we show that leveraging uncertainty in a smoothed classifier helps out-of-distribution detection.



## **37. Classification under strategic adversary manipulation using pessimistic bilevel optimisation**

cs.LG

27 pages, 5 figures, under review

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20284v1) [paper-pdf](http://arxiv.org/pdf/2410.20284v1)

**Authors**: David Benfield, Stefano Coniglio, Martin Kunc, Phan Tu Vuong, Alain Zemkoho

**Abstract**: Adversarial machine learning concerns situations in which learners face attacks from active adversaries. Such scenarios arise in applications such as spam email filtering, malware detection and fake-image generation, where security methods must be actively updated to keep up with the ever improving generation of malicious data.We model these interactions between the learner and the adversary as a game and formulate the problem as a pessimistic bilevel optimisation problem with the learner taking the role of the leader. The adversary, modelled as a stochastic data generator, takes the role of the follower, generating data in response to the classifier. While existing models rely on the assumption that the adversary will choose the least costly solution leading to a convex lower-level problem with a unique solution, we present a novel model and solution method which do not make such assumptions. We compare these to the existing approach and see significant improvements in performance suggesting that relaxing these assumptions leads to a more realistic model.



## **38. CodePurify: Defend Backdoor Attacks on Neural Code Models via Entropy-based Purification**

cs.CR

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20136v1) [paper-pdf](http://arxiv.org/pdf/2410.20136v1)

**Authors**: Fangwen Mu, Junjie Wang, Zhuohao Yu, Lin Shi, Song Wang, Mingyang Li, Qing Wang

**Abstract**: Neural code models have found widespread success in tasks pertaining to code intelligence, yet they are vulnerable to backdoor attacks, where an adversary can manipulate the victim model's behavior by inserting triggers into the source code. Recent studies indicate that advanced backdoor attacks can achieve nearly 100% attack success rates on many software engineering tasks. However, effective defense techniques against such attacks remain insufficiently explored. In this study, we propose CodePurify, a novel defense against backdoor attacks on code models through entropy-based purification. Entropy-based purification involves the process of precisely detecting and eliminating the possible triggers in the source code while preserving its semantic information. Within this process, CodePurify first develops a confidence-driven entropy-based measurement to determine whether a code snippet is poisoned and, if so, locates the triggers. Subsequently, it purifies the code by substituting the triggers with benign tokens using a masked language model. We extensively evaluate CodePurify against four advanced backdoor attacks across three representative tasks and two popular code models. The results show that CodePurify significantly outperforms four commonly used defense baselines, improving average defense performance by at least 40%, 40%, and 12% across the three tasks, respectively. These findings highlight the potential of CodePurify to serve as a robust defense against backdoor attacks on neural code models.



## **39. Adversarial Attacks Against Double RIS-Assisted MIMO Systems-based Autoencoder in Finite-Scattering Environments**

cs.IT

5 pages, 2 figures. Accepted by WCL

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20103v1) [paper-pdf](http://arxiv.org/pdf/2410.20103v1)

**Authors**: Bui Duc Son, Ngo Nam Khanh, Trinh Van Chien, Dong In Kim

**Abstract**: Autoencoder permits the end-to-end optimization and design of wireless communication systems to be more beneficial than traditional signal processing. However, this emerging learning-based framework has weaknesses, especially sensitivity to physical attacks. This paper explores adversarial attacks against a double reconfigurable intelligent surface (RIS)-assisted multiple-input and multiple-output (MIMO)-based autoencoder, where an adversary employs encoded and decoded datasets to create adversarial perturbation and fool the system. Because of the complex and dynamic data structures, adversarial attacks are not unique, each having its own benefits. We, therefore, propose three algorithms generating adversarial examples and perturbations to attack the RIS-MIMO-based autoencoder, exploiting the gradient descent and allowing for flexibility via varying the input dimensions. Numerical results show that the proposed adversarial attack-based algorithm significantly degrades the system performance regarding the symbol error rate compared to the jamming attacks.



## **40. Generative Adversarial Patches for Physical Attacks on Cross-Modal Pedestrian Re-Identification**

cs.CV

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20097v1) [paper-pdf](http://arxiv.org/pdf/2410.20097v1)

**Authors**: Yue Su, Hao Li, Maoguo Gong

**Abstract**: Visible-infrared pedestrian Re-identification (VI-ReID) aims to match pedestrian images captured by infrared cameras and visible cameras. However, VI-ReID, like other traditional cross-modal image matching tasks, poses significant challenges due to its human-centered nature. This is evidenced by the shortcomings of existing methods, which struggle to extract common features across modalities, while losing valuable information when bridging the gap between them in the implicit feature space, potentially compromising security. To address this vulnerability, this paper introduces the first physical adversarial attack against VI-ReID models. Our method, termed Edge-Attack, specifically tests the models' ability to leverage deep-level implicit features by focusing on edge information, the most salient explicit feature differentiating individuals across modalities. Edge-Attack utilizes a novel two-step approach. First, a multi-level edge feature extractor is trained in a self-supervised manner to capture discriminative edge representations for each individual. Second, a generative model based on Vision Transformer Generative Adversarial Networks (ViTGAN) is employed to generate adversarial patches conditioned on the extracted edge features. By applying these patches to pedestrian clothing, we create realistic, physically-realizable adversarial samples. This black-box, self-supervised approach ensures the generalizability of our attack against various VI-ReID models. Extensive experiments on SYSU-MM01 and RegDB datasets, including real-world deployments, demonstrate the effectiveness of Edge- Attack in significantly degrading the performance of state-of-the-art VI-ReID methods.



## **41. Attacks against Abstractive Text Summarization Models through Lead Bias and Influence Functions**

cs.CL

10 pages, 3 figures, Accepted at EMNLP Findings 2024

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20019v1) [paper-pdf](http://arxiv.org/pdf/2410.20019v1)

**Authors**: Poojitha Thota, Shirin Nilizadeh

**Abstract**: Large Language Models have introduced novel opportunities for text comprehension and generation. Yet, they are vulnerable to adversarial perturbations and data poisoning attacks, particularly in tasks like text classification and translation. However, the adversarial robustness of abstractive text summarization models remains less explored. In this work, we unveil a novel approach by exploiting the inherent lead bias in summarization models, to perform adversarial perturbations. Furthermore, we introduce an innovative application of influence functions, to execute data poisoning, which compromises the model's integrity. This approach not only shows a skew in the models behavior to produce desired outcomes but also shows a new behavioral change, where models under attack tend to generate extractive summaries rather than abstractive summaries.



## **42. Backdoor in Seconds: Unlocking Vulnerabilities in Large Pre-trained Models via Model Editing**

cs.AI

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.18267v2) [paper-pdf](http://arxiv.org/pdf/2410.18267v2)

**Authors**: Dongliang Guo, Mengxuan Hu, Zihan Guan, Junfeng Guo, Thomas Hartvigsen, Sheng Li

**Abstract**: Large pre-trained models have achieved notable success across a range of downstream tasks. However, recent research shows that a type of adversarial attack ($\textit{i.e.,}$ backdoor attack) can manipulate the behavior of machine learning models through contaminating their training dataset, posing significant threat in the real-world application of large pre-trained model, especially for those customized models. Therefore, addressing the unique challenges for exploring vulnerability of pre-trained models is of paramount importance. Through empirical studies on the capability for performing backdoor attack in large pre-trained models ($\textit{e.g.,}$ ViT), we find the following unique challenges of attacking large pre-trained models: 1) the inability to manipulate or even access large training datasets, and 2) the substantial computational resources required for training or fine-tuning these models. To address these challenges, we establish new standards for an effective and feasible backdoor attack in the context of large pre-trained models. In line with these standards, we introduce our EDT model, an \textbf{E}fficient, \textbf{D}ata-free, \textbf{T}raining-free backdoor attack method. Inspired by model editing techniques, EDT injects an editing-based lightweight codebook into the backdoor of large pre-trained models, which replaces the embedding of the poisoned image with the target image without poisoning the training dataset or training the victim model. Our experiments, conducted across various pre-trained models such as ViT, CLIP, BLIP, and stable diffusion, and on downstream tasks including image classification, image captioning, and image generation, demonstrate the effectiveness of our method. Our code is available in the supplementary material.



## **43. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

cs.CL

18 pages

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.18469v2) [paper-pdf](http://arxiv.org/pdf/2410.18469v2)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99% ASR on GPT-3.5 and 49% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.



## **44. RobustKV: Defending Large Language Models against Jailbreak Attacks via KV Eviction**

cs.CR

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19937v1) [paper-pdf](http://arxiv.org/pdf/2410.19937v1)

**Authors**: Tanqiu Jiang, Zian Wang, Jiacheng Liang, Changjiang Li, Yuhui Wang, Ting Wang

**Abstract**: Jailbreak attacks circumvent LLMs' built-in safeguards by concealing harmful queries within jailbreak prompts. While existing defenses primarily focus on mitigating the effects of jailbreak prompts, they often prove inadequate as jailbreak prompts can take arbitrary, adaptive forms. This paper presents RobustKV, a novel defense that adopts a fundamentally different approach by selectively removing critical tokens of harmful queries from key-value (KV) caches. Intuitively, for a jailbreak prompt to be effective, its tokens must achieve sufficient `importance' (as measured by attention scores), which inevitably lowers the importance of tokens in the concealed harmful query. Thus, by strategically evicting the KVs of the lowest-ranked tokens, RobustKV diminishes the presence of the harmful query in the KV cache, thus preventing the LLM from generating malicious responses. Extensive evaluation using benchmark datasets and models demonstrates that RobustKV effectively counters state-of-the-art jailbreak attacks while maintaining the LLM's general performance on benign queries. Moreover, RobustKV creates an intriguing evasiveness dilemma for adversaries, forcing them to balance between evading RobustKV and bypassing the LLM's built-in safeguards. This trade-off contributes to RobustKV's robustness against adaptive attacks. (warning: this paper contains potentially harmful content generated by LLMs.)



## **45. Robust Thompson Sampling Algorithms Against Reward Poisoning Attacks**

cs.LG

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19705v1) [paper-pdf](http://arxiv.org/pdf/2410.19705v1)

**Authors**: Yinglun Xu, Zhiwei Wang, Gagandeep Singh

**Abstract**: Thompson sampling is one of the most popular learning algorithms for online sequential decision-making problems and has rich real-world applications. However, current Thompson sampling algorithms are limited by the assumption that the rewards received are uncorrupted, which may not be true in real-world applications where adversarial reward poisoning exists. To make Thompson sampling more reliable, we want to make it robust against adversarial reward poisoning. The main challenge is that one can no longer compute the actual posteriors for the true reward, as the agent can only observe the rewards after corruption. In this work, we solve this problem by computing pseudo-posteriors that are less likely to be manipulated by the attack. We propose robust algorithms based on Thompson sampling for the popular stochastic and contextual linear bandit settings in both cases where the agent is aware or unaware of the budget of the attacker. We theoretically show that our algorithms guarantee near-optimal regret under any attack strategy.



## **46. A constrained optimization approach to improve robustness of neural networks**

cs.LG

29 pages, 4 figures, 5 tables

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2409.13770v2) [paper-pdf](http://arxiv.org/pdf/2409.13770v2)

**Authors**: Shudian Zhao, Jan Kronqvist

**Abstract**: In this paper, we present a novel nonlinear programming-based approach to fine-tune pre-trained neural networks to improve robustness against adversarial attacks while maintaining high accuracy on clean data. Our method introduces adversary-correction constraints to ensure correct classification of adversarial data and minimizes changes to the model parameters. We propose an efficient cutting-plane-based algorithm to iteratively solve the large-scale nonconvex optimization problem by approximating the feasible region through polyhedral cuts and balancing between robustness and accuracy. Computational experiments on standard datasets such as MNIST and CIFAR10 demonstrate that the proposed approach significantly improves robustness, even with a very small set of adversarial data, while maintaining minimal impact on accuracy.



## **47. Detecting adversarial attacks on random samples**

math.PR

title changed; introduction expanded; new results about spherical  attacks

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2408.06166v2) [paper-pdf](http://arxiv.org/pdf/2408.06166v2)

**Authors**: Gleb Smirnov

**Abstract**: This paper studies the problem of detecting adversarial perturbations in a sequence of observations. Given a data sample $X_1, \ldots, X_n$ drawn from a standard normal distribution, an adversary, after observing the sample, can perturb each observation by a fixed magnitude or leave it unchanged. We explore the relationship between the perturbation magnitude, the sparsity of the perturbation, and the detectability of the adversary's actions, establishing precise thresholds for when detection becomes impossible.



## **48. Corpus Poisoning via Approximate Greedy Gradient Descent**

cs.IR

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2406.05087v2) [paper-pdf](http://arxiv.org/pdf/2406.05087v2)

**Authors**: Jinyan Su, Preslav Nakov, Claire Cardie

**Abstract**: Dense retrievers are widely used in information retrieval and have also been successfully extended to other knowledge intensive areas such as language models, e.g., Retrieval-Augmented Generation (RAG) systems. Unfortunately, they have recently been shown to be vulnerable to corpus poisoning attacks in which a malicious user injects a small fraction of adversarial passages into the retrieval corpus to trick the system into returning these passages among the top-ranked results for a broad set of user queries. Further study is needed to understand the extent to which these attacks could limit the deployment of dense retrievers in real-world applications. In this work, we propose Approximate Greedy Gradient Descent (AGGD), a new attack on dense retrieval systems based on the widely used HotFlip method for efficiently generating adversarial passages. We demonstrate that AGGD can select a higher quality set of token-level perturbations than HotFlip by replacing its random token sampling with a more structured search. Experimentally, we show that our method achieves a high attack success rate on several datasets and using several retrievers, and can generalize to unseen queries and new domains. Notably, our method is extremely effective in attacking the ANCE retrieval model, achieving attack success rates that are 15.24\% and 17.44\% higher on the NQ and MS MARCO datasets, respectively, compared to HotFlip. Additionally, we demonstrate AGGD's potential to replace HotFlip in other adversarial attacks, such as knowledge poisoning of RAG systems.



## **49. Adversarial Attacks on Large Language Models Using Regularized Relaxation**

cs.LG

8 pages, 6 figures

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.19160v1) [paper-pdf](http://arxiv.org/pdf/2410.19160v1)

**Authors**: Samuel Jacob Chacko, Sajib Biswas, Chashi Mahiul Islam, Fatema Tabassum Liza, Xiuwen Liu

**Abstract**: As powerful Large Language Models (LLMs) are now widely used for numerous practical applications, their safety is of critical importance. While alignment techniques have significantly improved overall safety, LLMs remain vulnerable to carefully crafted adversarial inputs. Consequently, adversarial attack methods are extensively used to study and understand these vulnerabilities. However, current attack methods face significant limitations. Those relying on optimizing discrete tokens suffer from limited efficiency, while continuous optimization techniques fail to generate valid tokens from the model's vocabulary, rendering them impractical for real-world applications. In this paper, we propose a novel technique for adversarial attacks that overcomes these limitations by leveraging regularized gradients with continuous optimization methods. Our approach is two orders of magnitude faster than the state-of-the-art greedy coordinate gradient-based method, significantly improving the attack success rate on aligned language models. Moreover, it generates valid tokens, addressing a fundamental limitation of existing continuous optimization methods. We demonstrate the effectiveness of our attack on five state-of-the-art LLMs using four datasets.



## **50. Provably Robust Watermarks for Open-Source Language Models**

cs.CR

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18861v1) [paper-pdf](http://arxiv.org/pdf/2410.18861v1)

**Authors**: Miranda Christ, Sam Gunn, Tal Malkin, Mariana Raykova

**Abstract**: The recent explosion of high-quality language models has necessitated new methods for identifying AI-generated text. Watermarking is a leading solution and could prove to be an essential tool in the age of generative AI. Existing approaches embed watermarks at inference and crucially rely on the large language model (LLM) specification and parameters being secret, which makes them inapplicable to the open-source setting. In this work, we introduce the first watermarking scheme for open-source LLMs. Our scheme works by modifying the parameters of the model, but the watermark can be detected from just the outputs of the model. Perhaps surprisingly, we prove that our watermarks are unremovable under certain assumptions about the adversary's knowledge. To demonstrate the behavior of our construction under concrete parameter instantiations, we present experimental results with OPT-6.7B and OPT-1.3B. We demonstrate robustness to both token substitution and perturbation of the model parameters. We find that the stronger of these attacks, the model-perturbation attack, requires deteriorating the quality score to 0 out of 100 in order to bring the detection rate down to 50%.



