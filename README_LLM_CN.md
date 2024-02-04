# Latest Large Language Model Attack Papers
**update at 2024-02-04 13:13:04**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks**

视觉-LLM可以通过自我生成的排版攻击来愚弄自己 cs.CV

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00626v1) [paper-pdf](http://arxiv.org/pdf/2402.00626v1)

**Authors**: Maan Qraitem, Nazia Tasnim, Kate Saenko, Bryan A. Plummer

**Abstract**: Recently, significant progress has been made on Large Vision-Language Models (LVLMs); a new class of VL models that make use of large pre-trained language models. Yet, their vulnerability to Typographic attacks, which involve superimposing misleading text onto an image remain unstudied. Furthermore, prior work typographic attacks rely on sampling a random misleading class from a predefined set of classes. However, the random chosen class might not be the most effective attack. To address these issues, we first introduce a novel benchmark uniquely designed to test LVLMs vulnerability to typographic attacks. Furthermore, we introduce a new and more effective typographic attack: Self-Generated typographic attacks. Indeed, our method, given an image, make use of the strong language capabilities of models like GPT-4V by simply prompting them to recommend a typographic attack. Using our novel benchmark, we uncover that typographic attacks represent a significant threat against LVLM(s). Furthermore, we uncover that typographic attacks recommended by GPT-4V using our new method are not only more effective against GPT-4V itself compared to prior work attacks, but also against a host of less capable yet popular open source models like LLaVA, InstructBLIP, and MiniGPT4.

摘要: 最近，在大视觉语言模型(LVLMS)方面取得了重大进展，这是一类利用大型预先训练的语言模型的新的视觉语言模型。然而，它们在排版攻击中的脆弱性仍未得到研究。排版攻击涉及在图像上叠加误导性文本。此外，以前的工作排版攻击依赖于从预定义的一组类中随机抽样误导类。然而，随机选择的类可能不是最有效的攻击。为了解决这些问题，我们首先引入了一个新的基准测试，该基准测试是专门为测试LVLMS对排版攻击的脆弱性而设计的。此外，我们引入了一种新的、更有效的排版攻击：自生成排版攻击。事实上，我们的方法，在给定图像的情况下，通过简单地提示他们推荐排版攻击，利用了GPT-4V等型号的强大语言能力。使用我们的新基准，我们发现排版攻击代表着对LVLM(S)的重大威胁。此外，我们发现GPT-4V使用我们的新方法推荐的排版攻击不仅对GPT-4V本身比以前的工作攻击更有效，而且对LLaVA、InstructBLIP和MiniGPT4等能力较差但流行的开源模型也更有效。



## **2. Hidding the Ghostwriters: An Adversarial Evaluation of AI-Generated Student Essay Detection**

隐藏代写人：人工智能生成的学生作文检测的对抗性评估 cs.CL

Accepted by EMNLP 2023 Main conference, Oral Presentation

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00412v1) [paper-pdf](http://arxiv.org/pdf/2402.00412v1)

**Authors**: Xinlin Peng, Ying Zhou, Ben He, Le Sun, Yingfei Sun

**Abstract**: Large language models (LLMs) have exhibited remarkable capabilities in text generation tasks. However, the utilization of these models carries inherent risks, including but not limited to plagiarism, the dissemination of fake news, and issues in educational exercises. Although several detectors have been proposed to address these concerns, their effectiveness against adversarial perturbations, specifically in the context of student essay writing, remains largely unexplored. This paper aims to bridge this gap by constructing AIG-ASAP, an AI-generated student essay dataset, employing a range of text perturbation methods that are expected to generate high-quality essays while evading detection. Through empirical experiments, we assess the performance of current AIGC detectors on the AIG-ASAP dataset. The results reveal that the existing detectors can be easily circumvented using straightforward automatic adversarial attacks. Specifically, we explore word substitution and sentence substitution perturbation methods that effectively evade detection while maintaining the quality of the generated essays. This highlights the urgent need for more accurate and robust methods to detect AI-generated student essays in the education domain.

摘要: 大型语言模型(LLM)在文本生成任务中表现出了非凡的能力。然而，利用这些模式存在固有的风险，包括但不限于抄袭、传播假新闻和教育练习中的问题。虽然已经提出了几种检测器来解决这些问题，但它们对抗对抗性干扰的有效性，特别是在学生作文中的有效性，在很大程度上还没有被探索。本文旨在通过构建人工智能生成的学生作文数据集AIG-ASAP来弥合这一差距，该数据库使用了一系列文本扰动方法，有望在避免检测的同时生成高质量的作文。通过实验，我们评估了现有的AIGC检测器在AIG-ASAP数据集上的性能。结果表明，现有的检测器可以很容易地通过直接的自动对抗性攻击来绕过。具体地说，我们探索了单词替换和句子替换扰动方法，这些方法在保持生成的论文质量的同时有效地躲避了检测。这突显出迫切需要更准确和更强大的方法来检测教育领域中人工智能生成的学生作文。



## **3. Safety of Multimodal Large Language Models on Images and Text**

多通道大语言模型在图像和文本上的安全性 cs.CV

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00357v1) [paper-pdf](http://arxiv.org/pdf/2402.00357v1)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Attracted by the impressive power of Multimodal Large Language Models (MLLMs), the public is increasingly utilizing them to improve the efficiency of daily work. Nonetheless, the vulnerabilities of MLLMs to unsafe instructions bring huge safety risks when these models are deployed in real-world scenarios. In this paper, we systematically survey current efforts on the evaluation, attack, and defense of MLLMs' safety on images and text. We begin with introducing the overview of MLLMs on images and text and understanding of safety, which helps researchers know the detailed scope of our survey. Then, we review the evaluation datasets and metrics for measuring the safety of MLLMs. Next, we comprehensively present attack and defense techniques related to MLLMs' safety. Finally, we analyze several unsolved issues and discuss promising research directions.

摘要: 被多模式大型语言模型(MLLMS)令人印象深刻的力量所吸引，公众越来越多地利用它们来提高日常工作效率。然而，当这些模型部署在现实世界的场景中时，MLLMS对不安全指令的脆弱性带来了巨大的安全风险。本文系统地综述了当前对MLLMS图像和文本安全性的评估、攻击和防御的研究进展。我们首先介绍MLLMS关于图像和文本的概述以及对安全性的理解，这有助于研究人员了解我们调查的详细范围。然后，我们回顾了用于衡量MLLMS安全性的评价数据集和度量。接下来，我们全面介绍了与MLLMS安全相关的攻防技术。最后，我们分析了一些尚未解决的问题，并讨论了未来的研究方向。



## **4. De-identification is not always enough**

消除身份认同并不总是足够的 cs.CL

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2402.00179v1) [paper-pdf](http://arxiv.org/pdf/2402.00179v1)

**Authors**: Atiquer Rahman Sarkar, Yao-Shun Chuang, Noman Mohammed, Xiaoqian Jiang

**Abstract**: For sharing privacy-sensitive data, de-identification is commonly regarded as adequate for safeguarding privacy. Synthetic data is also being considered as a privacy-preserving alternative. Recent successes with numerical and tabular data generative models and the breakthroughs in large generative language models raise the question of whether synthetically generated clinical notes could be a viable alternative to real notes for research purposes. In this work, we demonstrated that (i) de-identification of real clinical notes does not protect records against a membership inference attack, (ii) proposed a novel approach to generate synthetic clinical notes using the current state-of-the-art large language models, (iii) evaluated the performance of the synthetically generated notes in a clinical domain task, and (iv) proposed a way to mount a membership inference attack where the target model is trained with synthetic data. We observed that when synthetically generated notes closely match the performance of real data, they also exhibit similar privacy concerns to the real data. Whether other approaches to synthetically generated clinical notes could offer better trade-offs and become a better alternative to sensitive real notes warrants further investigation.

摘要: 对于共享隐私敏感数据，消除身份识别通常被认为足以保护隐私。合成数据也被认为是一种保护隐私的选择。最近数字和表格数据生成模型的成功以及大型生成语言模型的突破提出了一个问题，即合成生成的临床笔记是否可以作为用于研究目的的真实笔记的可行替代方案。在这项工作中，我们证明了(I)真实临床笔记的去识别并不能保护记录免受成员关系推理攻击，(Ii)提出了一种使用当前最先进的大型语言模型生成合成临床笔记的新方法，(Iii)评估了合成生成的笔记在临床领域任务中的性能，以及(Iv)提出了一种利用合成数据训练目标模型的成员关系推理攻击的方法。我们观察到，当合成的笔记与真实数据的性能非常匹配时，它们也表现出与真实数据相似的隐私问题。合成临床笔记的其他方法是否可以提供更好的权衡，并成为敏感的真实笔记的更好替代方案，值得进一步研究。



## **5. LoRec: Large Language Model for Robust Sequential Recommendation against Poisoning Attacks**

LoRec：一种抗中毒攻击的鲁棒顺序推荐大语言模型 cs.IR

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2401.17723v1) [paper-pdf](http://arxiv.org/pdf/2401.17723v1)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Sequential recommender systems stand out for their ability to capture users' dynamic interests and the patterns of item-to-item transitions. However, the inherent openness of sequential recommender systems renders them vulnerable to poisoning attacks, where fraudulent users are injected into the training data to manipulate learned patterns. Traditional defense strategies predominantly depend on predefined assumptions or rules extracted from specific known attacks, limiting their generalizability to unknown attack types. To solve the above problems, considering the rich open-world knowledge encapsulated in Large Language Models (LLMs), our research initially focuses on the capabilities of LLMs in the detection of unknown fraudulent activities within recommender systems, a strategy we denote as LLM4Dec. Empirical evaluations demonstrate the substantial capability of LLMs in identifying unknown fraudsters, leveraging their expansive, open-world knowledge.   Building upon this, we propose the integration of LLMs into defense strategies to extend their effectiveness beyond the confines of known attacks. We propose LoRec, an advanced framework that employs LLM-Enhanced Calibration to strengthen the robustness of sequential recommender systems against poisoning attacks. LoRec integrates an LLM-enhanced CalibraTor (LCT) that refines the training process of sequential recommender systems with knowledge derived from LLMs, applying a user-wise reweighting to diminish the impact of fraudsters injected by attacks. By incorporating LLMs' open-world knowledge, the LCT effectively converts the limited, specific priors or rules into a more general pattern of fraudsters, offering improved defenses against poisoning attacks. Our comprehensive experiments validate that LoRec, as a general framework, significantly strengthens the robustness of sequential recommender systems.

摘要: 顺序推荐系统因其能够捕获用户的动态兴趣和项到项转换的模式而脱颖而出。然而，顺序推荐系统固有的开放性使得它们容易受到中毒攻击，在这种攻击中，欺诈性用户被注入到训练数据中以操纵学习模式。传统的防御策略主要依赖于从特定已知攻击中提取的预定义假设或规则，将其泛化为未知攻击类型。为了解决上述问题，考虑到大型语言模型(LLMS)中封装的丰富的开放世界知识，我们的研究最初集中在LLMS对推荐系统中未知欺诈活动的检测能力，我们将其命名为LLM4Dec。经验评估表明，LLMS利用其广博的、开放的知识，在识别未知欺诈者方面具有很强的能力。在此基础上，我们建议将LLM整合到防御战略中，以将其有效性扩展到已知攻击的范围之外。我们提出了LoRec，这是一个先进的框架，它使用LLM增强的校准来增强序列推荐系统对中毒攻击的健壮性。LoRec集成了LLM增强的校准器(LCT)，该校准器利用来自LLMS的知识来优化顺序推荐系统的训练过程，应用用户级的重新加权来减少攻击注入的欺诈者的影响。通过融入LLMS的开放世界知识，LCT有效地将有限的、特定的先例或规则转换为更一般的欺诈者模式，提供更好的防御中毒攻击的能力。我们的综合实验证明，LoRec作为一个通用的框架，显著增强了序列推荐系统的健壮性。



## **6. Unified Physical-Digital Face Attack Detection**

统一的物理-数字人脸攻击检测 cs.CV

12 pages, 8 figures

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2401.17699v1) [paper-pdf](http://arxiv.org/pdf/2401.17699v1)

**Authors**: Hao Fang, Ajian Liu, Haocheng Yuan, Junze Zheng, Dingheng Zeng, Yanhong Liu, Jiankang Deng, Sergio Escalera, Xiaoming Liu, Jun Wan, Zhen Lei

**Abstract**: Face Recognition (FR) systems can suffer from physical (i.e., print photo) and digital (i.e., DeepFake) attacks. However, previous related work rarely considers both situations at the same time. This implies the deployment of multiple models and thus more computational burden. The main reasons for this lack of an integrated model are caused by two factors: (1) The lack of a dataset including both physical and digital attacks with ID consistency which means the same ID covers the real face and all attack types; (2) Given the large intra-class variance between these two attacks, it is difficult to learn a compact feature space to detect both attacks simultaneously. To address these issues, we collect a Unified physical-digital Attack dataset, called UniAttackData. The dataset consists of $1,800$ participations of 2 and 12 physical and digital attacks, respectively, resulting in a total of 29,706 videos. Then, we propose a Unified Attack Detection framework based on Vision-Language Models (VLMs), namely UniAttackDetection, which includes three main modules: the Teacher-Student Prompts (TSP) module, focused on acquiring unified and specific knowledge respectively; the Unified Knowledge Mining (UKM) module, designed to capture a comprehensive feature space; and the Sample-Level Prompt Interaction (SLPI) module, aimed at grasping sample-level semantics. These three modules seamlessly form a robust unified attack detection framework. Extensive experiments on UniAttackData and three other datasets demonstrate the superiority of our approach for unified face attack detection.

摘要: 人脸识别(FR)系统可能会受到物理(即，打印照片)和数字(即，DeepFake)攻击。然而，以往的相关工作很少同时考虑这两种情况。这意味着需要部署多个模型，从而增加计算负担。缺乏综合模型的主要原因有两个：(1)缺乏包含物理攻击和数字攻击的具有ID一致性的数据集，这意味着相同的ID覆盖了真实人脸和所有攻击类型；(2)由于这两种攻击之间的类内差异很大，很难学习一个紧凑的特征空间来同时检测这两种攻击。为了解决这些问题，我们收集了一个统一的物理-数字攻击数据集，称为UniAttackData。该数据集包括2次和12次物理和数字攻击的1,800美元参与，总共产生29,706个视频。然后，提出了基于视觉语言模型(VLMS)的统一攻击检测框架UniAttack Detect，该框架包括三个主要模块：教师-学生提示(TSP)模块，用于获取统一的和特定的知识；统一知识挖掘(UKM)模块，用于捕获全面的特征空间；以及样本级提示交互(SLPI)模块，旨在获取样本级语义。这三个模块无缝地构成了一个强大的统一攻击检测框架。在UniAttackData和其他三个数据集上的大量实验证明了该方法在统一人脸攻击检测方面的优越性。



## **7. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的从弱到强的越狱 cs.CL

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17256v1) [paper-pdf](http://arxiv.org/pdf/2401.17256v1)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Although significant efforts have been dedicated to aligning large language models (LLMs), red-teaming reports suggest that these carefully aligned LLMs could still be jailbroken through adversarial prompts, tuning, or decoding. Upon examining the jailbreaking vulnerability of aligned LLMs, we observe that the decoding distributions of jailbroken and aligned models differ only in the initial generations. This observation motivates us to propose the weak-to-strong jailbreaking attack, where adversaries can utilize smaller unsafe/aligned LLMs (e.g., 7B) to guide jailbreaking against significantly larger aligned LLMs (e.g., 70B). To jailbreak, one only needs to additionally decode two smaller LLMs once, which involves minimal computation and latency compared to decoding the larger LLMs. The efficacy of this attack is demonstrated through experiments conducted on five models from three different organizations. Our study reveals a previously unnoticed yet efficient way of jailbreaking, exposing an urgent safety issue that needs to be considered when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 尽管已经致力于调整大型语言模型(LLM)，但红队报告表明，这些精心调整的LLM仍然可能通过敌意提示、调整或解码而越狱。在考察了对齐LLM的越狱脆弱性后，我们观察到，越狱模型和对齐模型的解码分布仅在最初几代中有所不同。这一观察结果促使我们提出了从弱到强的越狱攻击，其中对手可以利用较小的不安全/对齐的LLM(例如，7B)来指导对较大的对齐的LLM(例如，70B)的越狱。要越狱，只需额外解码两个较小的LLM一次，与解码较大的LLM相比，这涉及的计算量和延迟最小。通过对来自三个不同组织的五个模型进行的实验，证明了该攻击的有效性。我们的研究揭示了一种以前未被注意但有效的越狱方式，暴露了一个紧急的安全问题，在调整LLM时需要考虑这个问题。作为最初的尝试，我们提出了一种防御战略来防御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上找到



## **8. Noise Contrastive Estimation-based Matching Framework for Low-Resource Security Attack Pattern Recognition**

基于噪声对比估计的低资源安全攻击模式识别匹配框架 cs.LG

accepted at EACL 2024, in ARR October 2023

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.10337v3) [paper-pdf](http://arxiv.org/pdf/2401.10337v3)

**Authors**: Tu Nguyen, Nedim Šrndić, Alexander Neth

**Abstract**: Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning process of the matching model despite constrained resources.

摘要: 战术、技术和程序(TTP)代表了网络安全领域中复杂的攻击模式，在文本知识库中进行了全面的描述。在网络安全写作中识别TTP(通常称为TTP映射)是一项重要且具有挑战性的任务。传统的学习方法通常针对经典的多类或多标签分类环境中的问题。由于大量的类别(即TTP)、标签分布的不可避免的偏斜性以及标签空间的复杂层次结构，这种设置阻碍了模型的学习能力。我们用一种不同的学习范式来描述这个问题，其中文本到TTP标签的分配取决于两者之间的直接语义相似性，从而降低了单独竞争大标签空间的复杂性。为此，我们提出了一种具有有效的基于采样的学习-比较机制的神经匹配体系结构，使得匹配模型的学习过程在资源受限的情况下得以实现。



## **9. Provably Robust Multi-bit Watermarking for AI-generated Text via Error Correction Code**

基于纠错码的可证明鲁棒的人工智能文本多比特水印 cs.CR

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.16820v1) [paper-pdf](http://arxiv.org/pdf/2401.16820v1)

**Authors**: Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, Jiaheng Zhang

**Abstract**: Large Language Models (LLMs) have been widely deployed for their remarkable capability to generate texts resembling human language. However, they could be misused by criminals to create deceptive content, such as fake news and phishing emails, which raises ethical concerns. Watermarking is a key technique to mitigate the misuse of LLMs, which embeds a watermark (e.g., a bit string) into a text generated by a LLM. Consequently, this enables the detection of texts generated by a LLM as well as the tracing of generated texts to a specific user. The major limitation of existing watermark techniques is that they cannot accurately or efficiently extract the watermark from a text, especially when the watermark is a long bit string. This key limitation impedes their deployment for real-world applications, e.g., tracing generated texts to a specific user.   This work introduces a novel watermarking method for LLM-generated text grounded in \textbf{error-correction codes} to address this challenge. We provide strong theoretical analysis, demonstrating that under bounded adversarial word/token edits (insertion, deletion, and substitution), our method can correctly extract watermarks, offering a provable robustness guarantee. This breakthrough is also evidenced by our extensive experimental results. The experiments show that our method substantially outperforms existing baselines in both accuracy and robustness on benchmark datasets. For instance, when embedding a bit string of length 12 into a 200-token generated text, our approach attains an impressive match rate of $98.4\%$, surpassing the performance of Yoo et al. (state-of-the-art baseline) at $85.6\%$. When subjected to a copy-paste attack involving the injection of 50 tokens to generated texts with 200 words, our method maintains a substantial match rate of $90.8\%$, while the match rate of Yoo et al. diminishes to below $65\%$.

摘要: 大型语言模型(LLM)因其生成类似人类语言的文本的非凡能力而被广泛使用。然而，它们可能被犯罪分子滥用来创造欺骗性内容，如假新闻和钓鱼电子邮件，这引发了伦理问题。水印是缓解LLMS误用的一项关键技术，它将水印(如比特串)嵌入到LLM生成的文本中。因此，这使得能够检测由LLM生成的文本以及将生成的文本跟踪到特定用户。现有水印技术的主要局限性是不能准确或高效地从文本中提取水印，特别是当水印是长比特串的时候。这一关键限制阻碍了它们在现实世界应用程序中的部署，例如，跟踪生成的文本到特定用户。为了解决这一问题，提出了一种新的基于文本纠错码的LLM文本水印方法。我们提供了强有力的理论分析，证明了在有界的敌意单词/令牌编辑(插入、删除和替换)下，我们的方法可以正确地提取水印，提供了可证明的健壮性保证。这一突破也被我们广泛的实验结果所证明。实验表明，在基准数据集上，我们的方法在准确率和稳健性方面都大大优于现有的基线。例如，当将长度为12的比特串嵌入到200个标记生成的文本中时，我们的方法获得了令人印象深刻的匹配率$98.4\$，超过了Yoo等人的性能。(最新基线)为85.6美元。在对200个单词的文本进行50个标记的复制粘贴攻击时，我们的方法保持了相当高的匹配率为90.8美元，而Yoo等人的匹配率是90.8美元。降至65美元以下。



## **10. A Cross-Language Investigation into Jailbreak Attacks in Large Language Models**

大型语言模型中越狱攻击的跨语言研究 cs.CR

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.16765v1) [paper-pdf](http://arxiv.org/pdf/2401.16765v1)

**Authors**: Jie Li, Yi Liu, Chongyang Liu, Ling Shi, Xiaoning Ren, Yaowen Zheng, Yang Liu, Yinxing Xue

**Abstract**: Large Language Models (LLMs) have become increasingly popular for their advanced text generation capabilities across various domains. However, like any software, they face security challenges, including the risk of 'jailbreak' attacks that manipulate LLMs to produce prohibited content. A particularly underexplored area is the Multilingual Jailbreak attack, where malicious questions are translated into various languages to evade safety filters. Currently, there is a lack of comprehensive empirical studies addressing this specific threat.   To address this research gap, we conducted an extensive empirical study on Multilingual Jailbreak attacks. We developed a novel semantic-preserving algorithm to create a multilingual jailbreak dataset and conducted an exhaustive evaluation on both widely-used open-source and commercial LLMs, including GPT-4 and LLaMa. Additionally, we performed interpretability analysis to uncover patterns in Multilingual Jailbreak attacks and implemented a fine-tuning mitigation method. Our findings reveal that our mitigation strategy significantly enhances model defense, reducing the attack success rate by 96.2%. This study provides valuable insights into understanding and mitigating Multilingual Jailbreak attacks.

摘要: 大型语言模型(LLM)因其跨不同领域的高级文本生成能力而变得越来越受欢迎。然而，与任何软件一样，它们也面临着安全挑战，包括“越狱”攻击的风险，即操纵LLM生成被禁内容的风险。一个特别未被开发的领域是多语言越狱攻击，恶意问题被翻译成各种语言以逃避安全过滤器。目前，缺乏针对这一具体威胁的全面实证研究。为了弥补这一研究空白，我们对多语言越狱攻击进行了广泛的实证研究。我们开发了一种新的语义保持算法来创建多语言越狱数据集，并对广泛使用的开源和商业LLMS进行了详尽的评估，包括GPT-4和Llama。此外，我们还进行了可解释性分析，以揭示多语言越狱攻击的模式，并实现了一种微调的缓解方法。我们的研究结果表明，我们的缓解策略显著增强了模型防御，使攻击成功率降低了96.2%。这项研究为理解和减轻多语言越狱攻击提供了有价值的见解。



## **11. Low-Resource Languages Jailbreak GPT-4**

低资源语言越狱GPT-4 cs.CL

NeurIPS Workshop on Socially Responsible Language Modelling Research  (SoLaR) 2023. Best Paper Award

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2310.02446v2) [paper-pdf](http://arxiv.org/pdf/2310.02446v2)

**Authors**: Zheng-Xin Yong, Cristina Menghini, Stephen H. Bach

**Abstract**: AI safety training and red-teaming of large language models (LLMs) are measures to mitigate the generation of unsafe content. Our work exposes the inherent cross-lingual vulnerability of these safety mechanisms, resulting from the linguistic inequality of safety training data, by successfully circumventing GPT-4's safeguard through translating unsafe English inputs into low-resource languages. On the AdvBenchmark, GPT-4 engages with the unsafe translated inputs and provides actionable items that can get the users towards their harmful goals 79% of the time, which is on par with or even surpassing state-of-the-art jailbreaking attacks. Other high-/mid-resource languages have significantly lower attack success rate, which suggests that the cross-lingual vulnerability mainly applies to low-resource languages. Previously, limited training on low-resource languages primarily affects speakers of those languages, causing technological disparities. However, our work highlights a crucial shift: this deficiency now poses a risk to all LLMs users. Publicly available translation APIs enable anyone to exploit LLMs' safety vulnerabilities. Therefore, our work calls for a more holistic red-teaming efforts to develop robust multilingual safeguards with wide language coverage.

摘要: AI安全培训和大型语言模型(LLM)的红团队是减少不安全内容生成的措施。我们的工作通过将不安全的英语输入翻译成低资源的语言，成功地绕过了GPT-4的S保障，暴露了这些安全机制固有的跨语言漏洞，这是由于安全培训数据的语言不平等造成的。在AdvBenchmark上，GPT-4与不安全的翻译输入接触，并提供可操作的项目，可以在79%的时间内引导用户实现他们的有害目标，这与最先进的越狱攻击不相上下，甚至超过了这一水平。其他高/中资源语言的攻击成功率明显较低，这表明跨语言漏洞主要适用于低资源语言。以前，关于低资源语言的培训有限，主要影响说这些语言的人，造成技术差距。然而，我们的工作突出了一个关键的转变：这一缺陷现在对所有LLMS用户构成了风险。公开提供的转换API使任何人都能够利用LLMS的安全漏洞。因此，我们的工作需要更全面的红队努力，以制定具有广泛语言覆盖面的强大的多语言保障措施。



## **12. L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks**

L-AutoDA：利用大型语言模型进行基于决策的自动对抗性攻击 cs.CR

Under Review of IJCNN 2024

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.15335v1) [paper-pdf](http://arxiv.org/pdf/2401.15335v1)

**Authors**: Ping Guo, Fei Liu, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: In the rapidly evolving field of machine learning, adversarial attacks present a significant challenge to model robustness and security. Decision-based attacks, which only require feedback on the decision of a model rather than detailed probabilities or scores, are particularly insidious and difficult to defend against. This work introduces L-AutoDA (Large Language Model-based Automated Decision-based Adversarial Attacks), a novel approach leveraging the generative capabilities of Large Language Models (LLMs) to automate the design of these attacks. By iteratively interacting with LLMs in an evolutionary framework, L-AutoDA automatically designs competitive attack algorithms efficiently without much human effort. We demonstrate the efficacy of L-AutoDA on CIFAR-10 dataset, showing significant improvements over baseline methods in both success rate and computational efficiency. Our findings underscore the potential of language models as tools for adversarial attack generation and highlight new avenues for the development of robust AI systems.

摘要: 在快速发展的机器学习领域，敌意攻击对模型的健壮性和安全性提出了重大挑战。基于决策的攻击，只需要对模型的决策进行反馈，而不需要详细的概率或分数，特别隐蔽，很难防御。本文介绍了L-AUTODA(基于大语言模型的自动决策对抗性攻击)，它是一种利用大语言模型的生成能力来自动化攻击设计的新方法。通过在进化框架中迭代地与LLM交互，L自动DA无需太多人力即可自动高效地设计竞争攻击算法。我们在CIFAR-10数据集上验证了L-AutoDA的有效性，在成功率和计算效率上都比基线方法有了显著的提高。我们的发现强调了语言模型作为对抗性攻击生成工具的潜力，并强调了开发健壮的人工智能系统的新途径。



## **13. Better Representations via Adversarial Training in Pre-Training: A Theoretical Perspective**

前训练中通过对抗性训练获得更好的表征：一个理论视角 cs.LG

To appear in AISTATS2024

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2401.15248v1) [paper-pdf](http://arxiv.org/pdf/2401.15248v1)

**Authors**: Yue Xing, Xiaofeng Lin, Qifan Song, Yi Xu, Belinda Zeng, Guang Cheng

**Abstract**: Pre-training is known to generate universal representations for downstream tasks in large-scale deep learning such as large language models. Existing literature, e.g., \cite{kim2020adversarial}, empirically observe that the downstream tasks can inherit the adversarial robustness of the pre-trained model. We provide theoretical justifications for this robustness inheritance phenomenon. Our theoretical results reveal that feature purification plays an important role in connecting the adversarial robustness of the pre-trained model and the downstream tasks in two-layer neural networks. Specifically, we show that (i) with adversarial training, each hidden node tends to pick only one (or a few) feature; (ii) without adversarial training, the hidden nodes can be vulnerable to attacks. This observation is valid for both supervised pre-training and contrastive learning. With purified nodes, it turns out that clean training is enough to achieve adversarial robustness in downstream tasks.

摘要: 众所周知，预训练可以为大规模深度学习(如大型语言模型)中的下游任务生成通用表示。现有文献，如{kim2020对抗)，从经验上观察到下游任务可以继承预训练模型的对抗健壮性。我们为这种健壮性继承现象提供了理论依据。我们的理论结果表明，在两层神经网络中，特征提纯在连接预训练模型的对抗健壮性和下游任务方面起着重要作用。具体地说，我们证明了(I)在对抗性训练下，每个隐藏节点往往只选择一个(或几个)特征；(Ii)在没有对抗性训练的情况下，隐藏节点可能容易受到攻击。这一观察结果对有监督的预训练和对比学习都是有效的。事实证明，在净化节点的情况下，干净的训练足以在下游任务中实现对抗健壮性。



## **14. A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**

大型语言模型(LLM)安全与隐私：好、坏、丑 cs.CR

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2312.02003v2) [paper-pdf](http://arxiv.org/pdf/2312.02003v2)

**Authors**: Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, Yue Zhang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and Bard, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes the papers into "The Good" (beneficial LLM applications), "The Bad" (offensive applications), and "The Ugly" (vulnerabilities of LLMs and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code security (code vulnerability detection) and data privacy (data confidentiality protection), outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, Research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs' potential to both bolster and jeopardize cybersecurity.

摘要: 大型语言模型（LLM），如ChatGPT和Bard，彻底改变了自然语言的理解和生成。他们拥有深刻的语言理解能力，类似人类的文本生成能力，上下文意识和强大的解决问题的能力，使他们在各个领域（例如，搜索引擎、客户支持、翻译）。与此同时，LLM也在安全社区中获得了吸引力，揭示了安全漏洞并展示了其在安全相关任务中的潜力。本文探讨了LLM与安全和隐私的交叉点。具体来说，我们调查了LLM如何积极影响安全和隐私，与其使用相关的潜在风险和威胁，以及LLM内的固有漏洞。通过全面的文献综述，本文将论文分为“好”（有益的LLM应用程序），“坏”（攻击性应用程序）和“丑陋”（LLM及其防御的漏洞）。我们有一些有趣的发现。例如，LLM已被证明可以增强代码安全性（代码漏洞检测）和数据隐私性（数据机密性保护），优于传统方法。然而，由于它们具有类似人类的推理能力，它们也可以用于各种攻击（特别是用户级攻击）。我们已经确定了需要进一步研究的领域。例如，模型和参数提取攻击的研究是有限的，往往是理论，LLM参数规模和机密性的阻碍。安全指令调优是最近的发展，需要更多的探索。我们希望我们的工作能够揭示LLM在支持和危害网络安全方面的潜力。



## **15. TrojFST: Embedding Trojans in Few-shot Prompt Tuning**

TrojFST：在少发快调中嵌入特洛伊木马 cs.LG

9 pages

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2312.10467v2) [paper-pdf](http://arxiv.org/pdf/2312.10467v2)

**Authors**: Mengxin Zheng, Jiaqi Xue, Xun Chen, YanShan Wang, Qian Lou, Lei Jiang

**Abstract**: Prompt-tuning has emerged as a highly effective approach for adapting a pre-trained language model (PLM) to handle new natural language processing tasks with limited input samples. However, the success of prompt-tuning has led to adversaries attempting backdoor attacks against this technique. Previous prompt-based backdoor attacks faced challenges when implemented through few-shot prompt-tuning, requiring either full-model fine-tuning or a large training dataset. We observe the difficulty in constructing a prompt-based backdoor using few-shot prompt-tuning, which involves freezing the PLM and tuning a soft prompt with a restricted set of input samples. This approach introduces an imbalanced poisoned dataset, making it susceptible to overfitting and lacking attention awareness. To address these challenges, we introduce TrojFST for backdoor attacks within the framework of few-shot prompt-tuning. TrojFST comprises three modules: balanced poison learning, selective token poisoning, and trojan-trigger attention. In comparison to previous prompt-based backdoor attacks, TrojFST demonstrates significant improvements, enhancing ASR $> 9\%$ and CDA by $> 4\%$ across various PLMs and a diverse set of downstream tasks.

摘要: 即时调优已经成为一种高效的方法，可以使预先训练的语言模型(PLM)在输入样本有限的情况下处理新的自然语言处理任务。然而，快速调整的成功导致了对手试图对此技术进行后门攻击。以前的基于提示的后门攻击在通过少量提示调整实施时面临挑战，需要全模型微调或大型训练数据集。我们注意到使用少量提示调优来构建基于提示的后门的困难，这涉及冻结PLM并使用受限的输入样本集来调优软提示。这种方法引入了一个不平衡的有毒数据集，使其容易过度拟合和缺乏注意力意识。为了应对这些挑战，我们引入了TrojFST，用于在少发提示调优的框架内进行后门攻击。TrojFST包括三个模块：均衡毒物学习、选择性令牌毒化和木马触发注意。与以前基于提示的后门攻击相比，TrojFST表现出显著的改进，在不同的PLM和不同的下游任务集上将ASR$>9\$和CDA提高了$>4\$。



## **16. Adaptive Text Watermark for Large Language Models**

适用于大型语言模型的自适应文本水印 cs.CL

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2401.13927v1) [paper-pdf](http://arxiv.org/pdf/2401.13927v1)

**Authors**: Yepeng Liu, Yuheng Bu

**Abstract**: The advancement of Large Language Models (LLMs) has led to increasing concerns about the misuse of AI-generated text, and watermarking for LLM-generated text has emerged as a potential solution. However, it is challenging to generate high-quality watermarked text while maintaining strong security, robustness, and the ability to detect watermarks without prior knowledge of the prompt or model. This paper proposes an adaptive watermarking strategy to address this problem. To improve the text quality and maintain robustness, we adaptively add watermarking to token distributions with high entropy measured using an auxiliary model and keep the low entropy token distributions untouched. For the sake of security and to further minimize the watermark's impact on text quality, instead of using a fixed green/red list generated from a random secret key, which can be vulnerable to decryption and forgery, we adaptively scale up the output logits in proportion based on the semantic embedding of previously generated text using a well designed semantic mapping model. Our experiments involving various LLMs demonstrate that our approach achieves comparable robustness performance to existing watermark methods. Additionally, the text generated by our method has perplexity comparable to that of \emph{un-watermarked} LLMs while maintaining security even under various attacks.

摘要: 大型语言模型(LLM)的发展引起了人们对人工智能生成文本滥用的日益关注，LLM生成文本的水印已经成为一种潜在的解决方案。然而，在不预先知道提示或模型的情况下，在保持强大的安全性、健壮性和检测水印的能力的同时，生成高质量的水印文本是具有挑战性的。针对这一问题，本文提出了一种自适应水印策略。为了提高文本质量和保持稳健性，我们在使用辅助模型测量的高熵的令牌分布上自适应地添加水印，而对低熵的令牌分布保持不变。为了保证安全性，并进一步减小水印对文本质量的影响，不使用由随机密钥生成的易被解密和伪造的固定绿/红列表，而是使用设计良好的语义映射模型，基于先前生成的文本的语义嵌入，按比例自适应地放大输出逻辑。实验表明，该方法具有与现有水印方法相当的稳健性。此外，该方法生成的文本具有与未加水印的LLMS相当的复杂性，同时即使在各种攻击下也能保持安全性。



## **17. TrojanPuzzle: Covertly Poisoning Code-Suggestion Models**

特洛伊木马之谜：秘密中毒代码-建议模型 cs.CR

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2301.02344v2) [paper-pdf](http://arxiv.org/pdf/2301.02344v2)

**Authors**: Hojjat Aghakhani, Wei Dai, Andre Manoel, Xavier Fernandes, Anant Kharkar, Christopher Kruegel, Giovanni Vigna, David Evans, Ben Zorn, Robert Sim

**Abstract**: With tools like GitHub Copilot, automatic code suggestion is no longer a dream in software engineering. These tools, based on large language models, are typically trained on massive corpora of code mined from unvetted public sources. As a result, these models are susceptible to data poisoning attacks where an adversary manipulates the model's training by injecting malicious data. Poisoning attacks could be designed to influence the model's suggestions at run time for chosen contexts, such as inducing the model into suggesting insecure code payloads. To achieve this, prior attacks explicitly inject the insecure code payload into the training data, making the poison data detectable by static analysis tools that can remove such malicious data from the training set. In this work, we demonstrate two novel attacks, COVERT and TROJANPUZZLE, that can bypass static analysis by planting malicious poison data in out-of-context regions such as docstrings. Our most novel attack, TROJANPUZZLE, goes one step further in generating less suspicious poison data by never explicitly including certain (suspicious) parts of the payload in the poison data, while still inducing a model that suggests the entire payload when completing code (i.e., outside docstrings). This makes TROJANPUZZLE robust against signature-based dataset-cleansing methods that can filter out suspicious sequences from the training data. Our evaluation against models of two sizes demonstrates that both COVERT and TROJANPUZZLE have significant implications for practitioners when selecting code used to train or tune code-suggestion models.

摘要: 有了GitHub Copilot这样的工具，自动代码建议不再是软件工程中的梦想。这些工具基于大型语言模型，通常针对从未经审查的公共来源挖掘的大量代码语料库进行培训。因此，这些模型容易受到数据中毒攻击，即对手通过注入恶意数据来操纵模型的训练。毒化攻击可以被设计成影响模型在运行时对所选上下文的建议，例如诱导模型建议不安全的代码有效负载。为了实现这一点，先前的攻击明确地将不安全的代码有效负载注入到训练数据中，使得有毒数据可以被静态分析工具检测到，该静态分析工具可以从训练集中移除此类恶意数据。在这项工作中，我们展示了两种新的攻击，COMERT和TROJANPUZLE，它们可以通过在文档字符串等脱离上下文的区域植入恶意毒物数据来绕过静态分析。我们最新颖的攻击TROJANPUZLE在生成不那么可疑的有毒数据方面更进一步，它从未显式地将某些(可疑)有效负载部分包括在有毒数据中，同时仍诱导出一个模型，该模型在完成代码(即，文档字符串外部)时建议整个有效负载。这使得TROJANPUZLE对于基于签名的数据集清理方法具有健壮性，这些方法可以从训练数据中过滤出可疑序列。我们对两种规模的模型的评估表明，CONVERT和TROJANPUZLE对于实践者在选择用于训练或调整代码建议模型的代码时都有重要的影响。



## **18. How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**

约翰尼如何说服低层管理人员越狱：通过将低层管理人员人性化来挑战人工智能安全的再思考 cs.CL

14 pages of the main text, qualitative examples of jailbreaks may be  harmful in nature

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.06373v2) [paper-pdf](http://arxiv.org/pdf/2401.06373v2)

**Authors**: Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi

**Abstract**: Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs

摘要: 大多数传统的人工智能安全研究都将人工智能模型视为机器，并集中在安全专家开发的以算法为重点的攻击上。随着大型语言模型(LLM)变得越来越普遍和有能力，非专家用户也可能在日常交互中带来风险。本文介绍了一种新的视角，将越狱LLMS作为类人类的沟通者，来探索日常语言交互和人工智能安全之间被忽视的交集。具体地说，我们研究如何说服LLMS越狱。首先，我们提出了一种源于数十年社会科学研究的说服分类法。然后，我们应用分类法自动生成可解释的说服性对抗性提示(PAP)来越狱LLM。结果表明，说服显著提高了所有风险类别的越狱性能：PAP在Llama 2-7b Chat、GPT-3.5和GPT-4上的攻击成功率在10美元的试验中始终保持在92美元以上，超过了最近针对算法的攻击。在防御方面，我们探索了各种对抗PAP和的机制，发现了现有防御措施中的显著差距，并倡导从更根本上缓解高度互动的LLM



## **19. Text Embedding Inversion Attacks on Multilingual Language Models**

多语言语言模型的文本嵌入逆向攻击 cs.CL

13 pages

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.12192v1) [paper-pdf](http://arxiv.org/pdf/2401.12192v1)

**Authors**: Yiyi Chen, Heather Lent, Johannes Bjerva

**Abstract**: Representing textual information as real-numbered embeddings has become the norm in NLP. Moreover, with the rise of public interest in large language models (LLMs), Embeddings as a Service (EaaS) has rapidly gained traction as a business model. This is not without outstanding security risks, as previous research has demonstrated that sensitive data can be reconstructed from embeddings, even without knowledge of the underlying model that generated them. However, such work is limited by its sole focus on English, leaving all other languages vulnerable to attacks by malicious actors. %As many international and multilingual companies leverage EaaS, there is an urgent need for research into multilingual LLM security. To this end, this work investigates LLM security from the perspective of multilingual embedding inversion. Concretely, we define the problem of black-box multilingual and cross-lingual inversion attacks, with special attention to a cross-domain scenario. Our findings reveal that multilingual models are potentially more vulnerable to inversion attacks than their monolingual counterparts. This stems from the reduced data requirements for achieving comparable inversion performance in settings where the underlying language is not known a-priori. To our knowledge, this work is the first to delve into multilinguality within the context of inversion attacks, and our findings highlight the need for further investigation and enhanced defenses in the area of NLP Security.

摘要: 将文本信息表示为实数嵌入已成为NLP的规范。此外，随着公众对大型语言模型（LLM）兴趣的增加，嵌入式即服务（EaaS）作为一种商业模式迅速获得了关注。这并非没有突出的安全风险，因为以前的研究已经证明，即使不知道生成它们的底层模型，也可以从嵌入中重建敏感数据。然而，这种工作仅限于英语，使所有其他语言都容易受到恶意行为者的攻击。由于许多国际和多语言公司利用EaaS，迫切需要研究多语言LLM安全性。为此，本文从多语言嵌入倒置的角度研究了LLM的安全性。具体地说，我们定义了黑盒多语言和跨语言反转攻击的问题，特别注意跨域的情况。我们的研究结果表明，多语言模型比单语言模型更容易受到反转攻击。这源于在基础语言先验未知的环境中实现可比反演性能所需的数据减少。据我们所知，这项工作是第一次在反转攻击的背景下深入研究多语言性，我们的发现强调了在NLP安全领域进一步调查和增强防御的必要性。



## **20. All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks**

万事俱备：越狱攻击的简单黑匣子方法 cs.CL

12 pages, 4 figures, 2 tables

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.09798v2) [paper-pdf](http://arxiv.org/pdf/2401.09798v2)

**Authors**: Kazuhiro Takemoto

**Abstract**: Large Language Models (LLMs) like ChatGPT face `jailbreak' challenges, where safeguards are bypassed to produce ethically harmful prompts. This study proposes a simple black-box method to effectively generate jailbreak prompts, overcoming the high complexity and computational costs associated with existing methods. The proposed technique iteratively rewrites harmful prompts into non-harmful expressions using the target LLM itself, based on the hypothesis that LLMs can directly sample expressions that bypass safeguards. Demonstrated through experiments with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, this method achieved an attack success rate of over 80% within an average of 5 iterations and remained effective despite model updates. The generated jailbreak prompts were naturally-worded and concise; moreover, they were difficult-to-defend. These results indicate that creating effective jailbreak prompts is simpler than previously considered, suggesting that black-box jailbreak attacks pose a more serious threat.

摘要: 像ChatGPT这样的大型语言模型(LLM)面临着“越狱”的挑战，在这种情况下，安全措施会被绕过，产生道德上有害的提示。这项研究提出了一种简单的黑盒方法来有效地生成越狱提示，克服了现有方法相关的高复杂性和计算成本。所提出的技术使用目标LLM本身迭代地将有害提示重写为无害的表达，该方法基于LLM可以直接采样绕过安全措施的表达的假设。通过对ChatGPT(GPT-3.5和GPT-4)和Gemini-Pro的实验证明，该方法在平均5次迭代内达到了80%以上的攻击成功率，并且在模型更新的情况下仍然有效。生成的越狱提示很自然，措辞简洁；而且，它们很难辩护。这些结果表明，创建有效的越狱提示比之前认为的要简单，这表明黑盒越狱攻击构成了更严重的威胁。



## **21. Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts**

通过系统提示的自我对抗性攻击越狱GPT-4V cs.CR

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2311.09127v2) [paper-pdf](http://arxiv.org/pdf/2311.09127v2)

**Authors**: Yuanwei Wu, Xiang Li, Yixin Liu, Pan Zhou, Lichao Sun

**Abstract**: Existing work on jailbreak Multimodal Large Language Models (MLLMs) has focused primarily on adversarial examples in model inputs, with less attention to vulnerabilities, especially in model API. To fill the research gap, we carry out the following work: 1) We discover a system prompt leakage vulnerability in GPT-4V. Through carefully designed dialogue, we successfully extract the internal system prompts of GPT-4V. This finding indicates potential exploitable security risks in MLLMs; 2) Based on the acquired system prompts, we propose a novel MLLM jailbreaking attack method termed SASP (Self-Adversarial Attack via System Prompt). By employing GPT-4 as a red teaming tool against itself, we aim to search for potential jailbreak prompts leveraging stolen system prompts. Furthermore, in pursuit of better performance, we also add human modification based on GPT-4's analysis, which further improves the attack success rate to 98.7\%; 3) We evaluated the effect of modifying system prompts to defend against jailbreaking attacks. Results show that appropriately designed system prompts can significantly reduce jailbreak success rates. Overall, our work provides new insights into enhancing MLLM security, demonstrating the important role of system prompts in jailbreaking. This finding could be leveraged to greatly facilitate jailbreak success rates while also holding the potential for defending against jailbreaks.

摘要: 现有关于越狱多模式大型语言模型(MLLMS)的工作主要集中在模型输入中的对抗性示例，对漏洞的关注较少，特别是在模型API中。为了填补这一研究空白，我们开展了以下工作：1)在GPT-4V中发现了一个系统即时泄漏漏洞。通过精心设计的对话，我们成功地提取了GPT-4V的内部系统提示。2)基于获得的系统提示，提出了一种新的基于系统提示的MLLM越狱攻击方法SASP(Self-Aversarial Attack by System Prompt)。通过使用GPT-4作为针对自己的红色团队工具，我们的目标是利用被盗的系统提示来搜索潜在的越狱提示。此外，为了追求更好的性能，我们还在GPT-4的S分析的基础上增加了人工修改，进一步将攻击成功率提高到98.7%。3)评估了修改系统提示对越狱攻击的防御效果。结果表明，设计适当的系统提示可以显著降低越狱成功率。总体而言，我们的工作为加强MLLM安全提供了新的见解，展示了系统提示在越狱中的重要作用。这一发现可以被用来极大地提高越狱成功率，同时还具有防御越狱的潜力。



## **22. Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks**

大型语言模型中的通用漏洞：情景学习后门攻击 cs.CL

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.05949v3) [paper-pdf](http://arxiv.org/pdf/2401.05949v3)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Unlike traditional fine-tuning methods, in-context learning adapts pre-trained models to unseen tasks without updating any parameters. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we have designed a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning prompts, which can make models behave in accordance with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 40B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models. Our findings highlight the vulnerabilities of language models, and we hope this work will raise awareness of the possible security threats associated with in-context learning.

摘要: 情境学习是一种弥合预训练和微调之间差距的范式，在几个NLP任务中表现出了很高的效率，特别是在少数情况下。与传统的微调方法不同，情景学习使预先训练的模型适应未知的任务，而不需要更新任何参数。尽管情景学习被广泛应用，但它很容易受到恶意攻击。在这项工作中，我们提出了对此范式的安全担忧。我们的研究表明，攻击者可以通过毒化演示上下文来操纵大型语言模型的行为，而不需要对模型进行微调。具体地说，我们设计了一种新的后门攻击方法ICLAttack，以基于上下文学习的大型语言模型为目标。我们的方法包括两种类型的攻击：中毒演示示例和中毒提示，这可以使模型的行为符合预定义的意图。ICLAttack不需要额外的微调来植入后门，从而保持了模型的通用性。此外，有毒的例子被正确地标记，增强了我们攻击方法的自然隐蔽性。在几个语言模型上的广泛实验结果，从1.3B到40B参数的大小，证明了我们的攻击方法的有效性，例如在OPT模型上的三个数据集上的高平均攻击成功率为95.0%。我们的发现突显了语言模型的脆弱性，我们希望这项工作将提高人们对与情景学习相关的潜在安全威胁的认识。



## **23. InferAligner: Inference-Time Alignment for Harmlessness through Cross-Model Guidance**

InferAligner：通过跨模型引导实现无害的推理-时间对齐 cs.CL

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11206v1) [paper-pdf](http://arxiv.org/pdf/2401.11206v1)

**Authors**: Pengyu Wang, Dong Zhang, Linyang Li, Chenkun Tan, Xinghao Wang, Ke Ren, Botian Jiang, Xipeng Qiu

**Abstract**: With the rapid development of large language models (LLMs), they are not only used as general-purpose AI assistants but are also customized through further fine-tuning to meet the requirements of different applications. A pivotal factor in the success of current LLMs is the alignment process. Current alignment methods, such as supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF), focus on training-time alignment and are often complex and cumbersome to implement. Therefore, we develop \textbf{InferAligner}, a novel inference-time alignment method that utilizes cross-model guidance for harmlessness alignment. InferAligner utilizes safety steering vectors extracted from safety-aligned model to modify the activations of the target model when responding to harmful inputs, thereby guiding the target model to provide harmless responses. Experimental results show that our method can be very effectively applied to domain-specific models in finance, medicine, and mathematics, as well as to multimodal large language models (MLLMs) such as LLaVA. It significantly diminishes the Attack Success Rate (ASR) of both harmful instructions and jailbreak attacks, while maintaining almost unchanged performance in downstream tasks.

摘要: 随着大语言模型(LLM)的快速发展，它们不仅被用作通用的AI助手，而且还通过进一步微调进行定制，以满足不同应用的需求。当前LLMS成功的一个关键因素是对准过程。当前的比对方法，如监督微调(SFT)和从人类反馈的强化学习(RLHF)，侧重于训练时间比对，而且实现起来往往复杂和繁琐。因此，我们开发了一种新的推理-时间比对方法--文本bf{InferAligner}，该方法利用跨模型指导进行无害比对。当响应有害输入时，InferAligner利用从安全对齐模型中提取的安全导向向量来修改目标模型的激活，从而引导目标模型提供无害的响应。实验结果表明，我们的方法可以非常有效地应用于金融、医学、数学等领域的特定模型，以及LLaVA等多通道大语言模型。它显著降低了有害指令和越狱攻击的攻击成功率(ASR)，同时在下游任务中保持了几乎不变的性能。



## **24. Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images**

利用冗长图像诱导大型视觉语言模型的高能量潜伏期 cs.CV

Accepted by ICLR 2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11170v1) [paper-pdf](http://arxiv.org/pdf/2401.11170v1)

**Authors**: Kuofeng Gao, Yang Bai, Jindong Gu, Shu-Tao Xia, Philip Torr, Zhifeng Li, Wei Liu

**Abstract**: Large vision-language models (VLMs) such as GPT-4 have achieved exceptional performance across various multi-modal tasks. However, the deployment of VLMs necessitates substantial energy consumption and computational resources. Once attackers maliciously induce high energy consumption and latency time (energy-latency cost) during inference of VLMs, it will exhaust computational resources. In this paper, we explore this attack surface about availability of VLMs and aim to induce high energy-latency cost during inference of VLMs. We find that high energy-latency cost during inference of VLMs can be manipulated by maximizing the length of generated sequences. To this end, we propose verbose images, with the goal of crafting an imperceptible perturbation to induce VLMs to generate long sentences during inference. Concretely, we design three loss objectives. First, a loss is proposed to delay the occurrence of end-of-sequence (EOS) token, where EOS token is a signal for VLMs to stop generating further tokens. Moreover, an uncertainty loss and a token diversity loss are proposed to increase the uncertainty over each generated token and the diversity among all tokens of the whole generated sequence, respectively, which can break output dependency at token-level and sequence-level. Furthermore, a temporal weight adjustment algorithm is proposed, which can effectively balance these losses. Extensive experiments demonstrate that our verbose images can increase the length of generated sequences by 7.87 times and 8.56 times compared to original images on MS-COCO and ImageNet datasets, which presents potential challenges for various applications. Our code is available at https://github.com/KuofengGao/Verbose_Images.

摘要: 大型视觉语言模型(VLM)，如GPT-4，已经在各种多模式任务中取得了出色的性能。然而，VLMS的部署需要大量的能源消耗和计算资源。一旦攻击者在VLMS的推理过程中恶意导致高能耗和高延迟时间(能量延迟成本)，就会耗尽计算资源。在本文中，我们探索了关于VLMS可用性的攻击面，目的是在VLMS的推理过程中引入高能量延迟代价。我们发现，可以通过最大化生成序列的长度来控制VLMS推理过程中的高能量延迟代价。为此，我们提出了冗长的图像，目的是在推理过程中制作一种不可察觉的扰动来诱导VLM生成长句子。具体而言，我们设计了三个损失目标。首先，提出了一种延迟序列结束(EOS)令牌发生的损失，其中EOS令牌是VLM停止生成更多令牌的信号。此外，还提出了一种不确定性损失和令牌分集损失，分别增加了每个生成令牌的不确定性和整个生成序列中所有令牌之间的多样性，从而打破了令牌级和序列级的输出相关性。在此基础上，提出了一种时间权值调整算法，可以有效地平衡这些损失。大量实验表明，在MS-Coco和ImageNet数据集上，与原始图像相比，我们的冗长图像可以使生成的序列长度分别增加7.87倍和8.56倍，这给各种应用带来了潜在的挑战。我们的代码可以在https://github.com/KuofengGao/Verbose_Images.上找到



## **25. BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models**

BadChain：大型语言模型的后门思想链编译 cs.CR

Accepted to ICLR2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.12242v1) [paper-pdf](http://arxiv.org/pdf/2401.12242v1)

**Authors**: Zhen Xiang, Fengqing Jiang, Zidi Xiong, Bhaskar Ramasubramanian, Radha Poovendran, Bo Li

**Abstract**: Large language models (LLMs) are shown to benefit from chain-of-thought (COT) prompting, particularly when tackling tasks that require systematic reasoning processes. On the other hand, COT prompting also poses new vulnerabilities in the form of backdoor attacks, wherein the model will output unintended malicious content under specific backdoor-triggered conditions during inference. Traditional methods for launching backdoor attacks involve either contaminating the training dataset with backdoored instances or directly manipulating the model parameters during deployment. However, these approaches are not practical for commercial LLMs that typically operate via API access. In this paper, we propose BadChain, the first backdoor attack against LLMs employing COT prompting, which does not require access to the training dataset or model parameters and imposes low computational overhead. BadChain leverages the inherent reasoning capabilities of LLMs by inserting a backdoor reasoning step into the sequence of reasoning steps of the model output, thereby altering the final response when a backdoor trigger exists in the query prompt. Empirically, we show the effectiveness of BadChain for two COT strategies across four LLMs (Llama2, GPT-3.5, PaLM2, and GPT-4) and six complex benchmark tasks encompassing arithmetic, commonsense, and symbolic reasoning. Moreover, we show that LLMs endowed with stronger reasoning capabilities exhibit higher susceptibility to BadChain, exemplified by a high average attack success rate of 97.0% across the six benchmark tasks on GPT-4. Finally, we propose two defenses based on shuffling and demonstrate their overall ineffectiveness against BadChain. Therefore, BadChain remains a severe threat to LLMs, underscoring the urgency for the development of robust and effective future defenses.

摘要: 大型语言模型(LLM)被证明受益于思想链(COT)提示，特别是在处理需要系统推理过程的任务时。另一方面，CoT提示也以后门攻击的形式提出了新的漏洞，在推理过程中，模型会在特定后门触发的条件下输出意外的恶意内容。发起后门攻击的传统方法要么用后门实例污染训练数据集，要么在部署期间直接操作模型参数。然而，这些方法对于通常通过API访问操作的商业LLM来说并不实用。在本文中，我们提出了BadChain，这是第一个使用COT提示的针对LLMS的后门攻击，它不需要访问训练数据集或模型参数，并且施加了较低的计算开销。BadChain通过将后门推理步骤插入到模型输出的推理步骤序列中，从而在查询提示中存在后门触发器时更改最终响应，从而利用LLMS的固有推理能力。经验上，我们展示了BadChain在四个LLM(Llama2、GPT-3.5、Palm2和GPT-4)和六个复杂基准任务(包括算术、常识和符号推理)的两个COT策略上的有效性。此外，我们发现推理能力越强的LLM对BadChain的敏感度越高，在GPT-4上的六个基准任务上的平均攻击成功率高达97.0%。最后，我们提出了两种基于洗牌的防御方案，并证明了它们对BadChain的整体无效。因此，BadChain仍然是对LLMS的严重威胁，突显了发展强有力和有效的未来防御的紧迫性。



## **26. Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning**

为了保护而修剪：在没有微调的情况下增加对准LLM的越狱阻力 cs.LG

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10862v1) [paper-pdf](http://arxiv.org/pdf/2401.10862v1)

**Authors**: Adib Hasan, Ileana Rugina, Alex Wang

**Abstract**: Large Language Models (LLMs) are vulnerable to `Jailbreaking' prompts, a type of attack that can coax these models into generating harmful and illegal content. In this paper, we show that pruning up to 20% of LLM parameters markedly increases their resistance to such attacks without additional training and without sacrificing their performance in standard benchmarks. Intriguingly, we discovered that the enhanced safety observed post-pruning correlates to the initial safety training level of the model, hinting that the effect of pruning could be more general and may hold for other LLM behaviors beyond safety. Additionally, we introduce a curated dataset of 225 harmful tasks across five categories, inserted into ten different Jailbreaking prompts, showing that pruning aids LLMs in concentrating attention on task-relevant tokens in jailbreaking prompts. Lastly, our experiments reveal that the prominent chat models, such as LLaMA-2 Chat, Vicuna, and Mistral Instruct exhibit high susceptibility to jailbreaking attacks, with some categories achieving nearly 70-100% success rate. These insights underline the potential of pruning as a generalizable approach for improving LLM safety, reliability, and potentially other desired behaviors.

摘要: 大型语言模型(LLM)容易受到“越狱”提示的攻击，这是一种可以诱骗这些模型生成有害和非法内容的攻击类型。在本文中，我们表明，剪枝高达20%的LLM参数显著提高了它们对此类攻击的抵抗力，而不需要额外的训练，也不会牺牲它们在标准基准测试中的性能。有趣的是，我们发现剪枝后观察到的增强的安全性与模型的初始安全训练水平相关，这暗示剪枝的效果可能更一般，可能适用于安全以外的其他LLM行为。此外，我们引入了一个精选的数据集，其中包含五个类别的225个有害任务，插入到十个不同的越狱提示中，表明修剪有助于LLM将注意力集中在越狱提示中与任务相关的标记上。最后，我们的实验表明，著名的聊天模型，如骆驼-2聊天、维库纳和米斯特拉尔指令，对越狱攻击表现出很高的敏感度，其中一些类别的成功率接近70%-100%。这些见解强调了修剪作为一种可推广的方法来提高LLM的安全性、可靠性和潜在的其他所需行为的潜力。



## **27. Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings**

大型语言模型横向鱼叉式网络钓鱼：大规模组织背景下的比较研究 cs.CR

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09727v1) [paper-pdf](http://arxiv.org/pdf/2401.09727v1)

**Authors**: Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nishant Vishwamitra, Peyman Najafirad

**Abstract**: The critical threat of phishing emails has been further exacerbated by the potential of LLMs to generate highly targeted, personalized, and automated spear phishing attacks. Two critical problems concerning LLM-facilitated phishing require further investigation: 1) Existing studies on lateral phishing lack specific examination of LLM integration for large-scale attacks targeting the entire organization, and 2) Current anti-phishing infrastructure, despite its extensive development, lacks the capability to prevent LLM-generated attacks, potentially impacting both employees and IT security incident management. However, the execution of such investigative studies necessitates a real-world environment, one that functions during regular business operations and mirrors the complexity of a large organizational infrastructure. This setting must also offer the flexibility required to facilitate a diverse array of experimental conditions, particularly the incorporation of phishing emails crafted by LLMs. This study is a pioneering exploration into the use of Large Language Models (LLMs) for the creation of targeted lateral phishing emails, targeting a large tier 1 university's operation and workforce of approximately 9,000 individuals over an 11-month period. It also evaluates the capability of email filtering infrastructure to detect such LLM-generated phishing attempts, providing insights into their effectiveness and identifying potential areas for improvement. Based on our findings, we propose machine learning-based detection techniques for such emails to detect LLM-generated phishing emails that were missed by the existing infrastructure, with an F1-score of 98.96.

摘要: 钓鱼电子邮件的严重威胁由于LLMS可能产生高度针对性、个性化和自动化的鱼叉式钓鱼攻击而进一步加剧。与LLM协助的网络钓鱼有关的两个关键问题需要进一步调查：1)现有的横向网络钓鱼研究缺乏针对针对整个组织的大规模攻击的LLM集成的具体检查；2)当前的反网络钓鱼基础设施尽管得到了广泛的发展，但缺乏阻止LLM生成的攻击的能力，这可能会影响员工和IT安全事件管理。然而，进行这样的调查研究需要一个真实的环境，一个在常规业务运营期间运作的环境，一个反映大型组织基础设施复杂性的环境。这种设置还必须提供所需的灵活性，以促进不同的实验条件，特别是纳入由LLMS精心编制的钓鱼电子邮件。这项研究是对使用大型语言模型(LLM)创建有针对性的横向网络钓鱼电子邮件的开创性探索，目标是一所大型一线大学的运营和员工队伍，在11个月的时间里大约有9000人。它还评估了电子邮件过滤基础设施检测此类LLM生成的网络钓鱼尝试的能力，提供了对其有效性的见解，并确定了潜在的改进领域。基于我们的发现，我们提出了基于机器学习的此类电子邮件检测技术，以检测现有基础设施错过的LLM生成的钓鱼电子邮件，F1得分为98.96。



## **28. MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance**

MLLM-保护器：在不损害性能的情况下确保MLLM的安全 cs.CR

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.02906v2) [paper-pdf](http://arxiv.org/pdf/2401.02906v2)

**Authors**: Renjie Pi, Tianyang Han, Yueqi Xie, Rui Pan, Qing Lian, Hanze Dong, Jipeng Zhang, Tong Zhang

**Abstract**: The deployment of multimodal large language models (MLLMs) has brought forth a unique vulnerability: susceptibility to malicious attacks through visual inputs. We delve into the novel challenge of defending MLLMs against such attacks. We discovered that images act as a "foreign language" that is not considered during alignment, which can make MLLMs prone to producing harmful responses. Unfortunately, unlike the discrete tokens considered in text-based LLMs, the continuous nature of image signals presents significant alignment challenges, which poses difficulty to thoroughly cover the possible scenarios. This vulnerability is exacerbated by the fact that open-source MLLMs are predominantly fine-tuned on limited image-text pairs that is much less than the extensive text-based pretraining corpus, which makes the MLLMs more prone to catastrophic forgetting of their original abilities during explicit alignment tuning. To tackle these challenges, we introduce MLLM-Protector, a plug-and-play strategy combining a lightweight harm detector and a response detoxifier. The harm detector's role is to identify potentially harmful outputs from the MLLM, while the detoxifier corrects these outputs to ensure the response stipulates to the safety standards. This approach effectively mitigates the risks posed by malicious visual inputs without compromising the model's overall performance. Our results demonstrate that MLLM-Protector offers a robust solution to a previously unaddressed aspect of MLLM security.

摘要: 多模式大型语言模型(MLLMS)的部署带来了一个独特的漏洞：通过视觉输入易受恶意攻击。我们深入研究了保护MLLMS免受此类攻击的新挑战。我们发现，图像是一种“外语”，在对齐过程中没有考虑到这一点，这可能会使MLLM容易产生有害的反应。不幸的是，与基于文本的LLMS中考虑的离散令牌不同，图像信号的连续性质带来了巨大的对齐挑战，这使得很难完全覆盖可能的场景。开源MLLMS主要在有限的图文对上进行微调，这比基于大量文本的预训练语料库要少得多，这使得MLLMS在显式对齐调整期间更容易灾难性地忘记其原始能力，从而加剧了这一漏洞。为了应对这些挑战，我们引入了MLLM-Protector，这是一种结合了轻型伤害探测器和响应解毒器的即插即用策略。危害检测器的作用是识别MLLM的潜在有害输出，而解毒器纠正这些输出，以确保响应符合安全标准。这种方法有效地降低了恶意视觉输入带来的风险，而不会影响模型的整体性能。我们的结果表明，MLLM-Protector为MLLM安全的一个以前未解决的方面提供了一个健壮的解决方案。



## **29. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

AttackEval：如何评估越狱攻击大型语言模型的有效性 cs.CL

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09002v1) [paper-pdf](http://arxiv.org/pdf/2401.09002v1)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.

摘要: 在我们的研究中，我们开创了一种新的方法来评估越狱攻击对大型语言模型(如GPT-4和LLaMa2)的有效性，不同于传统的专注于健壮性的二进制评估。我们的研究引入了两个不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供了一个独特的视角，能够对攻击效果进行更全面和细微的评估，并使攻击者能够更好地了解他们的攻击提示。此外，我们还开发了专门为越狱任务量身定做的全面地面事实数据集。这一数据集不仅是我们当前研究的重要基准，而且还为未来的研究奠定了基础资源，使这一不断发展的领域能够进行一致和比较的分析。通过与传统评估方法的细致比较，我们发现我们的评估符合基线的趋势，同时提供了更深入和详细的评估。我们相信，通过准确评估越狱任务中攻击提示的有效性，我们的工作为评估快速注射领域中更广泛的类似甚至更复杂的任务奠定了坚实的基础，这可能会给这一领域带来革命性的变化。



## **30. Whispering Pixels: Exploiting Uninitialized Register Accesses in Modern GPUs**

低语像素：利用现代GPU中未初始化的寄存器访问 cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08881v1) [paper-pdf](http://arxiv.org/pdf/2401.08881v1)

**Authors**: Frederik Dermot Pustelnik, Xhani Marvin Saß, Jean-Pierre Seifert

**Abstract**: Graphic Processing Units (GPUs) have transcended their traditional use-case of rendering graphics and nowadays also serve as a powerful platform for accelerating ubiquitous, non-graphical rendering tasks. One prominent task is inference of neural networks, which process vast amounts of personal data, such as audio, text or images. Thus, GPUs became integral components for handling vast amounts of potentially confidential data, which has awakened the interest of security researchers. This lead to the discovery of various vulnerabilities in GPUs in recent years. In this paper, we uncover yet another vulnerability class in GPUs: We found that some GPU implementations lack proper register initialization routines before shader execution, leading to unintended register content leakage of previously executed shader kernels. We showcase the existence of the aforementioned vulnerability on products of 3 major vendors - Apple, NVIDIA and Qualcomm. The vulnerability poses unique challenges to an adversary due to opaque scheduling and register remapping algorithms present in the GPU firmware, complicating the reconstruction of leaked data. In order to illustrate the real-world impact of this flaw, we showcase how these challenges can be solved for attacking various workloads on the GPU. First, we showcase how uninitialized registers leak arbitrary pixel data processed by fragment shaders. We further implement information leakage attacks on intermediate data of Convolutional Neural Networks (CNNs) and present the attack's capability to leak and reconstruct the output of Large Language Models (LLMs).

摘要: 图形处理单元(GPU)已经超越了渲染图形的传统用例，如今也成为加速无处不在的非图形渲染任务的强大平台。一项突出的任务是神经网络的推理，它处理大量的个人数据，如音频、文本或图像。因此，GPU成为处理海量潜在机密数据的不可或缺的组件，这唤醒了安全研究人员的兴趣。这导致了近年来GPU中各种漏洞的发现。在本文中，我们发现了GPU中的另一个漏洞类别：我们发现一些GPU实现在着色器执行之前缺乏适当的寄存器初始化例程，导致先前执行的着色器内核的意外寄存器内容泄漏。我们展示了3家主要供应商的产品上存在上述漏洞-苹果、NVIDIA和高通。由于GPU固件中存在不透明的调度和寄存器重新映射算法，该漏洞对对手构成了独特的挑战，使泄漏数据的重建复杂化。为了说明该漏洞的实际影响，我们展示了如何解决这些挑战来攻击GPU上的各种工作负载。首先，我们展示了未初始化的寄存器如何泄漏由片段着色器处理的任意像素数据。在此基础上，对卷积神经网络(CNN)的中间数据进行了信息泄漏攻击，并给出了该攻击对大语言模型(LLM)输出的泄漏和重构能力。



## **31. IsamasRed: A Public Dataset Tracking Reddit Discussions on Israel-Hamas Conflict**

IsamasRed：一个追踪Reddit关于以色列和哈马斯冲突的公共数据集 cs.SI

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08202v1) [paper-pdf](http://arxiv.org/pdf/2401.08202v1)

**Authors**: Kai Chen, Zihao He, Keith Burghardt, Jingxin Zhang, Kristina Lerman

**Abstract**: The conflict between Israel and Palestinians significantly escalated after the October 7, 2023 Hamas attack, capturing global attention. To understand the public discourse on this conflict, we present a meticulously compiled dataset--IsamasRed--comprising nearly 400,000 conversations and over 8 million comments from Reddit, spanning from August 2023 to November 2023. We introduce an innovative keyword extraction framework leveraging a large language model to effectively identify pertinent keywords, ensuring a comprehensive data collection. Our initial analysis on the dataset, examining topics, controversy, emotional and moral language trends over time, highlights the emotionally charged and complex nature of the discourse. This dataset aims to enrich the understanding of online discussions, shedding light on the complex interplay between ideology, sentiment, and community engagement in digital spaces.

摘要: 2023年10月7日哈马斯袭击事件后，以色列和巴勒斯坦之间的冲突显著升级，引起了全球的关注。为了理解公众对这场冲突的讨论，我们提供了一个精心编制的数据集-IsamasRed--从2023年8月到2023年11月，包含来自Reddit的近40万次对话和800多万条评论。我们引入了一个创新的关键字提取框架，利用一个大型语言模型来有效地识别相关关键字，确保全面的数据收集。我们对数据集的初步分析，考察了话题、争议、情感和道德语言随着时间的推移的趋势，突显了话语的情绪化和复杂性。该数据集旨在丰富对在线讨论的理解，揭示数字空间中意识形态、情绪和社区参与之间的复杂相互作用。



## **32. MGTBench: Benchmarking Machine-Generated Text Detection**

MGTB：机器生成文本检测的基准测试 cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2303.14822v3) [paper-pdf](http://arxiv.org/pdf/2303.14822v3)

**Authors**: Xinlei He, Xinyue Shen, Zeyuan Chen, Michael Backes, Yang Zhang

**Abstract**: Nowadays, powerful large language models (LLMs) such as ChatGPT have demonstrated revolutionary power in a variety of tasks. Consequently, the detection of machine-generated texts (MGTs) is becoming increasingly crucial as LLMs become more advanced and prevalent. These models have the ability to generate human-like language, making it challenging to discern whether a text is authored by a human or a machine. This raises concerns regarding authenticity, accountability, and potential bias. However, existing methods for detecting MGTs are evaluated using different model architectures, datasets, and experimental settings, resulting in a lack of a comprehensive evaluation framework that encompasses various methodologies. Furthermore, it remains unclear how existing detection methods would perform against powerful LLMs. In this paper, we fill this gap by proposing the first benchmark framework for MGT detection against powerful LLMs, named MGTBench. Extensive evaluations on public datasets with curated texts generated by various powerful LLMs such as ChatGPT-turbo and Claude demonstrate the effectiveness of different detection methods. Our ablation study shows that a larger number of words in general leads to better performance and most detection methods can achieve similar performance with much fewer training samples. Moreover, we delve into a more challenging task: text attribution. Our findings indicate that the model-based detection methods still perform well in the text attribution task. To investigate the robustness of different detection methods, we consider three adversarial attacks, namely paraphrasing, random spacing, and adversarial perturbations. We discover that these attacks can significantly diminish detection effectiveness, underscoring the critical need for the development of more robust detection methods.

摘要: 如今，强大的大型语言模型(LLM)，如ChatGPT，已经在各种任务中展示了革命性的力量。因此，随着LLMS变得更加先进和普遍，机器生成文本(MGTS)的检测变得越来越重要。这些模型具有生成类似人类的语言的能力，这使得辨别文本是由人还是由机器创作具有挑战性。这引发了人们对真实性、问责性和潜在偏见的担忧。然而，现有的检测MGTS的方法是使用不同的模型体系结构、数据集和实验设置来评估的，导致缺乏包含各种方法的全面评估框架。此外，目前尚不清楚现有的检测方法将如何对抗强大的LLMS。在本文中，我们通过提出第一个针对强大的LLMS的MGT检测的基准框架来填补这一空白，称为MGTB。在公共数据集上的广泛评估与各种强大的LLMS生成的精选文本，如ChatGPT-Turbo和Claude证明了不同检测方法的有效性。我们的烧蚀研究表明，通常情况下，单词数量越多，性能越好，大多数检测方法都可以在更少的训练样本下获得类似的性能。此外，我们还深入研究了一项更具挑战性的任务：文本归因。我们的研究结果表明，基于模型的检测方法在文本归因任务中仍然表现良好。为了考察不同检测方法的稳健性，我们考虑了三种对抗性攻击，即释义攻击、随机间隔攻击和对抗性扰动攻击。我们发现，这些攻击会显著降低检测效率，这突显了开发更健壮的检测方法的迫切需要。



## **33. Traces of Memorisation in Large Language Models for Code**

代码的大型语言模型中的并行化痕迹 cs.CR

ICSE 2024 Research Track

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2312.11658v2) [paper-pdf](http://arxiv.org/pdf/2312.11658v2)

**Authors**: Ali Al-Kaswan, Maliheh Izadi, Arie van Deursen

**Abstract**: Large language models have gained significant popularity because of their ability to generate human-like text and potential applications in various fields, such as Software Engineering. Large language models for code are commonly trained on large unsanitised corpora of source code scraped from the internet. The content of these datasets is memorised and can be extracted by attackers with data extraction attacks. In this work, we explore memorisation in large language models for code and compare the rate of memorisation with large language models trained on natural language. We adopt an existing benchmark for natural language and construct a benchmark for code by identifying samples that are vulnerable to attack. We run both benchmarks against a variety of models, and perform a data extraction attack. We find that large language models for code are vulnerable to data extraction attacks, like their natural language counterparts. From the training data that was identified to be potentially extractable we were able to extract 47% from a CodeGen-Mono-16B code completion model. We also observe that models memorise more, as their parameter count grows, and that their pre-training data are also vulnerable to attack. We also find that data carriers are memorised at a higher rate than regular code or documentation and that different model architectures memorise different samples. Data leakage has severe outcomes, so we urge the research community to further investigate the extent of this phenomenon using a wider range of models and extraction techniques in order to build safeguards to mitigate this issue.

摘要: 大型语言模型由于其生成类人文本的能力以及在软件工程等各个领域的潜在应用而受到广泛欢迎。代码的大型语言模型通常是在从互联网上抓取的大型未经清理的源代码语料库上训练的。这些数据集的内容是记忆的，可以由攻击者通过数据提取攻击来提取。在这项工作中，我们探索了大型语言模型中的代码记忆，并将记忆率与自然语言训练的大型语言模型进行了比较。我们采用现有的自然语言基准，并通过识别易受攻击的样本来构建代码基准。我们针对各种模型运行这两个基准测试，并执行数据提取攻击。我们发现，代码的大型语言模型很容易受到数据提取攻击，就像它们的自然语言模型一样。从被识别为潜在可提取的训练数据中，我们能够从CodeGen-Mono-16 B代码完成模型中提取47%。我们还观察到，随着参数数量的增加，模型的记忆力会增加，而且它们的预训练数据也容易受到攻击。我们还发现，数据载体的记忆率高于常规代码或文档，不同的模型架构记忆不同的样本。数据泄露会产生严重的后果，因此我们敦促研究界使用更广泛的模型和提取技术进一步调查这一现象的程度，以建立保障措施来缓解这一问题。



## **34. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

多语种机器文本检测中的作者身份混淆 cs.CL

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07867v1) [paper-pdf](http://arxiv.org/pdf/2401.07867v1)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of latest Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause detection evasion in all tested languages, where homoglyph attacks are especially successful.

摘要: 最新的大型语言模型(LLM)的高质量文本生成能力引起了人们对它们的滥用(例如，在大规模生成/传播虚假信息中)的担忧。机器生成文本(MGT)检测对于应对此类威胁非常重要。然而，它容易受到作者身份混淆(AO)方法的影响，例如转译，这可能导致MGTS逃避检测。到目前为止，这只在单一语言环境中进行了评估。因此，最近提出的多语言检测器的敏感性仍然未知。我们通过全面基准测试10种著名的AO方法的性能来填补这一空白，针对11种语言的MGT攻击37种MGT检测方法(即，10$\乘以$37$\乘以$11=4,070个组合)。我们还使用混淆文本来评估数据增强对对手健壮性的影响。结果表明，所有测试的声学方法都能在所有测试的语言中造成检测逃避，其中同形文字攻击尤其成功。



## **35. Signed-Prompt: A New Approach to Prevent Prompt Injection Attacks Against LLM-Integrated Applications**

Signed-Prompt：一种防止LLM集成应用的即时注入攻击的新方法 cs.CR

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07612v1) [paper-pdf](http://arxiv.org/pdf/2401.07612v1)

**Authors**: Xuchen Suo

**Abstract**: The critical challenge of prompt injection attacks in Large Language Models (LLMs) integrated applications, a growing concern in the Artificial Intelligence (AI) field. Such attacks, which manipulate LLMs through natural language inputs, pose a significant threat to the security of these applications. Traditional defense strategies, including output and input filtering, as well as delimiter use, have proven inadequate. This paper introduces the 'Signed-Prompt' method as a novel solution. The study involves signing sensitive instructions within command segments by authorized users, enabling the LLM to discern trusted instruction sources. The paper presents a comprehensive analysis of prompt injection attack patterns, followed by a detailed explanation of the Signed-Prompt concept, including its basic architecture and implementation through both prompt engineering and fine-tuning of LLMs. Experiments demonstrate the effectiveness of the Signed-Prompt method, showing substantial resistance to various types of prompt injection attacks, thus validating its potential as a robust defense strategy in AI security.

摘要: 大型语言模型(LLMS)集成应用中的即时注入攻击的关键挑战，越来越受到人工智能(AI)领域的关注。这类攻击通过自然语言输入来操纵LLM，对这些应用程序的安全构成了重大威胁。事实证明，包括输出和输入过滤以及分隔符使用在内的传统防御策略是不够的。本文提出了一种新的解决方案--“签名提示法”。这项研究涉及授权用户在命令段内签署敏感指令，使LLM能够识别可信的指令来源。本文首先对即时注入攻击模式进行了全面的分析，然后详细说明了签名提示的概念，包括其基本结构和通过即时工程和微调LLMS实现的方法。实验证明了签名提示方法的有效性，对各种类型的即时注入攻击表现出很强的抵抗力，从而验证了其作为人工智能安全中一种健壮防御策略的潜力。



## **36. Stability Analysis of ChatGPT-based Sentiment Analysis in AI Quality Assurance**

人工智能质量保证中基于ChatGPT情感分析的稳定性分析 cs.CL

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07441v1) [paper-pdf](http://arxiv.org/pdf/2401.07441v1)

**Authors**: Tinghui Ouyang, AprilPyone MaungMaung, Koichi Konishi, Yoshiki Seo, Isao Echizen

**Abstract**: In the era of large AI models, the complex architecture and vast parameters present substantial challenges for effective AI quality management (AIQM), e.g. large language model (LLM). This paper focuses on investigating the quality assurance of a specific LLM-based AI product--a ChatGPT-based sentiment analysis system. The study delves into stability issues related to both the operation and robustness of the expansive AI model on which ChatGPT is based. Experimental analysis is conducted using benchmark datasets for sentiment analysis. The results reveal that the constructed ChatGPT-based sentiment analysis system exhibits uncertainty, which is attributed to various operational factors. It demonstrated that the system also exhibits stability issues in handling conventional small text attacks involving robustness.

摘要: 在大型人工智能模型的时代，复杂的体系结构和庞大的参数对有效的人工智能质量管理(AIQM)提出了巨大的挑战，例如大型语言模型(LLM)。本文重点研究了一个具体的基于LLM的人工智能产品--基于ChatGPT的情感分析系统的质量保证。这项研究深入探讨了与ChatGPT所基于的扩展人工智能模型的操作和健壮性相关的稳定性问题。使用用于情感分析的基准数据集进行了实验分析。结果表明，构建的基于ChatGPT的情感分析系统存在不确定性，这归因于各种操作因素。结果表明，该系统在处理涉及健壮性的常规小文本攻击时也存在稳定性问题。



## **37. Advancing TTP Analysis: Harnessing the Power of Encoder-Only and Decoder-Only Language Models with Retrieval Augmented Generation**

高级TTP分析：利用仅编码者和仅解码者的语言模型和检索增强生成的能力 cs.CR

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.00280v2) [paper-pdf](http://arxiv.org/pdf/2401.00280v2)

**Authors**: Reza Fayyazi, Rozhina Taghdimi, Shanchieh Jay Yang

**Abstract**: Tactics, Techniques, and Procedures (TTPs) outline the methods attackers use to exploit vulnerabilities. The interpretation of TTPs in the MITRE ATT&CK framework can be challenging for cybersecurity practitioners due to presumed expertise, complex dependencies, and inherent ambiguity. Meanwhile, advancements with Large Language Models (LLMs) have led to recent surge in studies exploring its uses in cybersecurity operations. This leads us to question how well encoder-only (e.g., RoBERTa) and decoder-only (e.g., GPT-3.5) LLMs can comprehend and summarize TTPs to inform analysts of the intended purposes (i.e., tactics) of a cyberattack procedure. The state-of-the-art LLMs have shown to be prone to hallucination by providing inaccurate information, which is problematic in critical domains like cybersecurity. Therefore, we propose the use of Retrieval Augmented Generation (RAG) techniques to extract relevant contexts for each cyberattack procedure for decoder-only LLMs (without fine-tuning). We further contrast such approach against supervised fine-tuning (SFT) of encoder-only LLMs. Our results reveal that both the direct-use of decoder-only LLMs (i.e., its pre-trained knowledge) and the SFT of encoder-only LLMs offer inaccurate interpretation of cyberattack procedures. Significant improvements are shown when RAG is used for decoder-only LLMs, particularly when directly relevant context is found. This study further sheds insights on the limitations and capabilities of using RAG for LLMs in interpreting TTPs.

摘要: 战术、技术和过程(TTP)概述了攻击者用来利用漏洞的方法。由于假定的专业知识、复杂的依赖关系和固有的模糊性，MITRE ATT&CK框架中对TTP的解释可能会对网络安全从业者构成挑战。与此同时，大型语言模型(LLM)的进步导致了最近探索其在网络安全行动中的应用的研究激增。这导致我们质疑仅编码器(例如Roberta)和仅解码器(例如GPT-3.5)的LLM能够在多大程度上理解和总结TTP以告知分析师网络攻击过程的预期目的(即战术)。最先进的LLM通过提供不准确的信息而容易产生幻觉，这在网络安全等关键领域是有问题的。因此，我们提出使用检索增强生成(RAG)技术来提取仅针对解码器的LLMS的每个网络攻击过程的相关上下文(无需微调)。我们进一步将这种方法与仅编码器的LLM的有监督微调(SFT)进行了对比。我们的结果表明，直接使用仅解码的LLM(即其预先训练的知识)和仅编码的LLM的SFT都提供了对网络攻击过程的不准确解释。当RAG被用于仅解码器的LLM时，尤其是当找到直接相关的上下文时，显示出显著的改进。这项研究进一步揭示了使用RAG对LLMS进行TTP解释的局限性和能力。



## **38. Intention Analysis Prompting Makes Large Language Models A Good Jailbreak Defender**

意图分析提示使大型语言模型成为优秀的越狱捍卫者 cs.CL

9 pages, 5 figures

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.06561v1) [paper-pdf](http://arxiv.org/pdf/2401.06561v1)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly in the face of stealthy and complex jailbreaks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis Prompting (IAPrompt). The principle behind is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, IAPrompt is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on SAP200 and DAN benchmarks across Vicuna, ChatGLM, MPT, DeepSeek, and GPT-3.5 show that IAPrompt could consistently and significantly reduce the harmfulness in response (averagely -46.5% attack success rate) and maintain the general helpfulness. Further analyses present some insights into how our method works. To facilitate reproducibility, We release our code and scripts at: https://github.com/alphadl/SafeLLM_with_IntentionAnalysis

摘要: 将大型语言模型（LLM）与人类价值观保持一致，特别是在面对隐秘而复杂的越狱时，这是一个艰巨的挑战。在这项研究中，我们提出了一个简单而高效的防御策略，即，意图分析验证（IApprompt）。其背后的原则是通过两个阶段的过程触发LLM固有的自我纠正和提高能力：1）基本意图分析，2）政策一致的反应。值得注意的是，IApprompt是一种仅推理的方法，因此可以提高LLM的安全性，而不会影响它们的有用性。在Vicuna、ChatGLM、MPT、DeepSeek和GPT-3.5上对SAP 200和DAN基准测试进行的广泛实验表明，IAPlmpt可以始终如一地显著降低响应中的危害性（平均攻击成功率为-46.5%），并保持一般的有用性。进一步的分析对我们的方法如何工作提出了一些见解。为了便于再现，我们在以下网站发布代码和脚本：https://github.com/alphadl/SafeLLM_with_IntentionAnalysis



## **39. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

朝向稳健剪枝：一种自适应的语言模型知识保留剪枝策略 cs.CL

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2310.13191v3) [paper-pdf](http://arxiv.org/pdf/2310.13191v3)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.

摘要: 修剪目标最近已经超越了语言模型中的精确度和稀疏性，扩展到了健壮性。尽管如此，现有的方法在不断增加模型稀疏性的同时努力增强对敌对攻击的鲁棒性，并且需要重新训练过程。随着人类步入大型语言模型时代，这些问题变得日益突出。本文提出语言模型的稳健性与它们所包含的预训练知识的程度成正比。因此，我们提出了一种训练后剪枝策略，旨在忠实地复制密集语言模型的嵌入空间和特征空间，目的是在剪枝过程中保存更多的预先训练的知识。在这种设置中，每一层的重建误差不仅源于自身，还包括来自前几层的累积误差，然后进行自适应校正。与其他最先进的基线相比，我们的方法在精确度、稀疏性、健壮性和剪枝成本之间表现出了更好的平衡，在数据集Sst2、IMDB和AgNews上使用ERT，标志着在语言模型中朝着健壮剪枝迈出了重要的一步。



## **40. LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack**

LimeAttack：文本硬标签对抗性攻击的局部可解释方法 cs.CL

18 pages, 38th AAAI Main Track

**SubmitDate**: 2024-01-10    [abs](http://arxiv.org/abs/2308.00319v2) [paper-pdf](http://arxiv.org/pdf/2308.00319v2)

**Authors**: Hai Zhu, Zhaoqing Yang, Weiwei Shang, Yuren Wu

**Abstract**: Natural language processing models are vulnerable to adversarial examples. Previous textual adversarial attacks adopt gradients or confidence scores to calculate word importance ranking and generate adversarial examples. However, this information is unavailable in the real world. Therefore, we focus on a more realistic and challenging setting, named hard-label attack, in which the attacker can only query the model and obtain a discrete prediction label. Existing hard-label attack algorithms tend to initialize adversarial examples by random substitution and then utilize complex heuristic algorithms to optimize the adversarial perturbation. These methods require a lot of model queries and the attack success rate is restricted by adversary initialization. In this paper, we propose a novel hard-label attack algorithm named LimeAttack, which leverages a local explainable method to approximate word importance ranking, and then adopts beam search to find the optimal solution. Extensive experiments show that LimeAttack achieves the better attacking performance compared with existing hard-label attack under the same query budget. In addition, we evaluate the effectiveness of LimeAttack on large language models, and results indicate that adversarial examples remain a significant threat to large language models. The adversarial examples crafted by LimeAttack are highly transferable and effectively improve model robustness in adversarial training.

摘要: 自然语言处理模型容易受到对抗性示例的影响。以前的文本对抗性攻击采用梯度或置信度来计算单词重要性排名并生成对抗性示例。然而，这些信息在现实世界中是不可用的。因此，我们专注于一个更现实和更具挑战性的设置，命名为硬标签攻击，其中攻击者只能查询模型并获得离散的预测标签。现有的硬标签攻击算法倾向于通过随机替换初始化对抗样本，然后利用复杂的启发式算法来优化对抗扰动。这些方法需要大量的模型查询，攻击成功率受到对手初始化的限制。在本文中，我们提出了一种新的硬标签攻击算法命名为LimeAttack，它利用本地可解释的方法来近似字的重要性排名，然后采用波束搜索找到最优解。实验结果表明，在相同的查询开销下，LimeAttack比现有的硬标签攻击具有更好的攻击性能。此外，我们评估了LimeAttack在大型语言模型上的有效性，结果表明对抗性示例仍然是大型语言模型的重大威胁。LimeAttack制作的对抗性示例具有高度可转移性，并有效提高了对抗性训练中的模型鲁棒性。



## **41. Jatmo: Prompt Injection Defense by Task-Specific Finetuning**

Jatmo：通过特定于任务的微调实现快速注入防御 cs.CR

24 pages, 6 figures

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2312.17673v2) [paper-pdf](http://arxiv.org/pdf/2312.17673v2)

**Authors**: Julien Piet, Maha Alrashed, Chawin Sitawarin, Sizhe Chen, Zeming Wei, Elizabeth Sun, Basel Alomair, David Wagner

**Abstract**: Large Language Models (LLMs) are attracting significant research attention due to their instruction-following abilities, allowing users and developers to leverage LLMs for a variety of tasks. However, LLMs are vulnerable to prompt-injection attacks: a class of attacks that hijack the model's instruction-following abilities, changing responses to prompts to undesired, possibly malicious ones. In this work, we introduce Jatmo, a method for generating task-specific models resilient to prompt-injection attacks. Jatmo leverages the fact that LLMs can only follow instructions once they have undergone instruction tuning. It harnesses a teacher instruction-tuned model to generate a task-specific dataset, which is then used to fine-tune a base model (i.e., a non-instruction-tuned model). Jatmo only needs a task prompt and a dataset of inputs for the task: it uses the teacher model to generate outputs. For situations with no pre-existing datasets, Jatmo can use a single example, or in some cases none at all, to produce a fully synthetic dataset. Our experiments on seven tasks show that Jatmo models provide similar quality of outputs on their specific task as standard LLMs, while being resilient to prompt injections. The best attacks succeeded in less than 0.5% of cases against our models, versus 87% success rate against GPT-3.5-Turbo. We release Jatmo at https://github.com/wagner-group/prompt-injection-defense.

摘要: 大型语言模型(LLM)由于其遵循指令的能力而吸引了大量的研究关注，使用户和开发人员能够利用LLM来执行各种任务。然而，LLM很容易受到即时注入攻击：这是一类劫持模型的指令遵循能力的攻击，将对提示的响应更改为不受欢迎的、可能是恶意的提示。在这项工作中，我们介绍了Jatmo，一种生成对快速注入攻击具有弹性的特定任务模型的方法。Jatmo利用了这样一个事实，即LLM只有在经过指令调优后才能遵循指令。它利用教师指令调整的模型来生成特定于任务的数据集，然后使用该数据集来微调基本模型(即，非指令调整的模型)。Jatmo只需要任务提示符和任务输入的数据集：它使用教师模型来生成输出。对于没有预先存在的数据集的情况，Jatmo可以使用单个示例，或者在某些情况下根本不使用任何示例来生成完全合成的数据集。我们在七个任务上的实验表明，Jatmo模型在其特定任务中提供的输出质量与标准LLM相似，同时对快速注入具有弹性。在针对我们的模型的情况下，最好的攻击成功率不到0.5%，而针对GPT-3.5-Turbo的成功率为87%。我们在https://github.com/wagner-group/prompt-injection-defense.发布了贾特莫



## **42. The Stronger the Diffusion Model, the Easier the Backdoor: Data Poisoning to Induce Copyright Breaches Without Adjusting Finetuning Pipeline**

扩散模型越强，后门就越容易：数据中毒在不调整微调管道的情况下引发版权侵犯 cs.CR

This study reveals that by subtly inserting non-copyright-infringing  poisoning data into a diffusion model's training dataset, it's possible to  trigger the model to generate copyrighted content, highlighting  vulnerabilities in current copyright protection strategies

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2401.04136v1) [paper-pdf](http://arxiv.org/pdf/2401.04136v1)

**Authors**: Haonan Wang, Qianli Shen, Yao Tong, Yang Zhang, Kenji Kawaguchi

**Abstract**: The commercialization of diffusion models, renowned for their ability to generate high-quality images that are often indistinguishable from real ones, brings forth potential copyright concerns. Although attempts have been made to impede unauthorized access to copyrighted material during training and to subsequently prevent DMs from generating copyrighted images, the effectiveness of these solutions remains unverified. This study explores the vulnerabilities associated with copyright protection in DMs by introducing a backdoor data poisoning attack (SilentBadDiffusion) against text-to-image diffusion models. Our attack method operates without requiring access to or control over the diffusion model's training or fine-tuning processes; it merely involves the insertion of poisoning data into the clean training dataset. This data, comprising poisoning images equipped with prompts, is generated by leveraging the powerful capabilities of multimodal large language models and text-guided image inpainting techniques. Our experimental results and analysis confirm the method's effectiveness. By integrating a minor portion of non-copyright-infringing stealthy poisoning data into the clean dataset-rendering it free from suspicion-we can prompt the finetuned diffusion models to produce copyrighted content when activated by specific trigger prompts. These findings underline potential pitfalls in the prevailing copyright protection strategies and underscore the necessity for increased scrutiny and preventative measures against the misuse of DMs.

摘要: 扩散模型的商业化带来了潜在的版权问题，这些模型以生成高质量图像的能力而闻名，而这些图像往往与真实图像难以区分。尽管已尝试在培训期间阻止未经授权访问受版权保护的材料，并随后阻止DM生成受版权保护的图像，但这些解决方案的有效性仍未得到证实。这项研究通过引入针对文本到图像扩散模型的后门数据中毒攻击(SilentBadDiffulation)来探索DM中与版权保护相关的漏洞。我们的攻击方法不需要访问或控制扩散模型的训练或微调过程；它只涉及将中毒数据插入到干净的训练数据集中。这些数据包括带有提示的中毒图像，是通过利用多模式大型语言模型和文本引导的图像修复技术的强大功能生成的。实验结果和分析证实了该方法的有效性。通过将一小部分非侵犯版权的隐形中毒数据集成到干净的数据集中-使其不受怀疑-我们可以在特定触发提示激活时提示精细调整的扩散模型生成受版权保护的内容。这些调查结果突显了现行版权保护战略中的潜在陷阱，并强调了加强审查和预防滥用数字版权管理的必要性。



## **43. PromptBench: A Unified Library for Evaluation of Large Language Models**

PromptBitch：大型语言模型评估的统一库 cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2312.07910v2) [paper-pdf](http://arxiv.org/pdf/2312.07910v2)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

摘要: 大型语言模型(LLM)的评估对于评估其性能和降低潜在的安全风险至关重要。在本文中，我们介绍了一个用于评估LLMS的统一库PromptBitch.它由几个易于研究人员使用和扩展的关键组件组成：即时构建、即时工程、数据集和模型加载、对抗性即时攻击、动态评估协议和分析工具。PromptBitch是一个开放的、通用的、灵活的研究代码库，可以在创建新的基准、部署下游应用程序和设计新的评估协议方面促进原创研究。该代码可在https://github.com/microsoft/promptbench上获得，并将继续受到支持。



## **44. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

InstructTA：针对大型视觉语言模型的指令调整定向攻击 cs.CV

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.01886v2) [paper-pdf](http://arxiv.org/pdf/2312.01886v2)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.

摘要: 大型视觉语言模型在图像理解和响应生成方面表现出了令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性例子的攻击。本文提出了一种新颖实用的灰盒攻击方案，即攻击者只能访问受害者LVLM的可视编码器，而不知道其提示(通常是服务提供商的专有提示，而不是公开可用的)及其底层的大型语言模型(LLM)。这一实际设置对目标对抗性攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种指令调谐的定向攻击(InstructTA)，对具有高可转移性的LVLMS进行定向对抗性攻击。首先，我们利用一个公开的文本到图像的生成模型将目标响应“反转”成目标图像，并使用GPT-4从目标响应中推断出合理的指令符号。然后，我们形成一个局部代理模型(与受害者LVLM共享相同的视觉编码器)来提取对抗性图像示例和目标图像的指令感知特征，并最小化这两个特征之间的距离以优化对抗性示例。为了进一步提高可转移性，我们用转译自LLM的指令扩充了指令$\boldSymbol{p}^\Prime$。大量实验证明了该方法在目标攻击性能和可转移性方面的优越性。



## **45. Mining Temporal Attack Patterns from Cyberthreat Intelligence Reports**

从网络威胁情报报告中挖掘时态攻击模式 cs.CR

A modified version of this pre-print is submitted to IEEE  Transactions on Software Engineering, and is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01883v1) [paper-pdf](http://arxiv.org/pdf/2401.01883v1)

**Authors**: Md Rayhanur Rahman, Brandon Wroblewski, Quinn Matthews, Brantley Morgan, Tim Menzies, Laurie Williams

**Abstract**: Defending from cyberattacks requires practitioners to operate on high-level adversary behavior. Cyberthreat intelligence (CTI) reports on past cyberattack incidents describe the chain of malicious actions with respect to time. To avoid repeating cyberattack incidents, practitioners must proactively identify and defend against recurring chain of actions - which we refer to as temporal attack patterns. Automatically mining the patterns among actions provides structured and actionable information on the adversary behavior of past cyberattacks. The goal of this paper is to aid security practitioners in prioritizing and proactive defense against cyberattacks by mining temporal attack patterns from cyberthreat intelligence reports. To this end, we propose ChronoCTI, an automated pipeline for mining temporal attack patterns from cyberthreat intelligence (CTI) reports of past cyberattacks. To construct ChronoCTI, we build the ground truth dataset of temporal attack patterns and apply state-of-the-art large language models, natural language processing, and machine learning techniques. We apply ChronoCTI on a set of 713 CTI reports, where we identify 124 temporal attack patterns - which we categorize into nine pattern categories. We identify that the most prevalent pattern category is to trick victim users into executing malicious code to initiate the attack, followed by bypassing the anti-malware system in the victim network. Based on the observed patterns, we advocate organizations to train users about cybersecurity best practices, introduce immutable operating systems with limited functionalities, and enforce multi-user authentications. Moreover, we advocate practitioners to leverage the automated mining capability of ChronoCTI and design countermeasures against the recurring attack patterns.

摘要: 防御网络攻击需要从业者对高级别的对手行为进行操作。关于过去网络攻击事件的网络威胁情报(CTI)报告描述了与时间相关的恶意行动链。为了避免重复网络攻击事件，从业人员必须主动识别和防御反复出现的动作链--我们将其称为临时攻击模式。自动挖掘动作之间的模式提供了关于过去网络攻击对手行为的结构化和可操作的信息。本文的目的是通过从网络威胁情报报告中挖掘时态攻击模式，帮助安全从业者确定优先顺序并主动防御网络攻击。为此，我们提出了ChronoCTI，这是一种从过去网络攻击的网络威胁情报(CTI)报告中挖掘时态攻击模式的自动化管道。为了构建ChronoCTI，我们建立了时间攻击模式的地面事实数据集，并应用了最先进的大型语言模型、自然语言处理和机器学习技术。我们在一组713个CTI报告上应用了ChronoCTI，其中我们识别了124个临时攻击模式-我们将其分类为9个模式类别。我们发现，最普遍的模式类别是诱骗受害者用户执行恶意代码来发起攻击，然后绕过受害者网络中的反恶意软件系统。根据观察到的模式，我们倡导组织对用户进行网络安全最佳实践方面的培训，引入具有有限功能的不变操作系统，并强制实施多用户身份验证。此外，我们提倡实践者利用ChronoCTI的自动挖掘能力，并设计针对反复出现的攻击模式的对策。



## **46. Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment**

安全和性能，为什么不能两者兼而有之呢？针对AI软件部署异构性攻击的双目标优化模型压缩 cs.AI

Accepted by IEEE Transactions on Software Engineering (TSE).  Camera-ready Version. arXiv admin note: substantial text overlap with  arXiv:2208.05969

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00996v1) [paper-pdf](http://arxiv.org/pdf/2401.00996v1)

**Authors**: Jie Zhu, Leye Wang, Xiao Han, Anmin Liu, Tao Xie

**Abstract**: The size of deep learning models in artificial intelligence (AI) software is increasing rapidly, hindering the large-scale deployment on resource-restricted devices (e.g., smartphones). To mitigate this issue, AI software compression plays a crucial role, which aims to compress model size while keeping high performance. However, the intrinsic defects in a big model may be inherited by the compressed one. Such defects may be easily leveraged by adversaries, since a compressed model is usually deployed in a large number of devices without adequate protection. In this article, we aim to address the safe model compression problem from the perspective of safety-performance co-optimization. Specifically, inspired by the test-driven development (TDD) paradigm in software engineering, we propose a test-driven sparse training framework called SafeCompress. By simulating the attack mechanism as safety testing, SafeCompress can automatically compress a big model to a small one following the dynamic sparse training paradigm. Then, considering two kinds of representative and heterogeneous attack mechanisms, i.e., black-box membership inference attack and white-box membership inference attack, we develop two concrete instances called BMIA-SafeCompress and WMIA-SafeCompress. Further, we implement another instance called MMIA-SafeCompress by extending SafeCompress to defend against the occasion when adversaries conduct black-box and white-box membership inference attacks simultaneously. We conduct extensive experiments on five datasets for both computer vision and natural language processing tasks. The results show the effectiveness and generalizability of our framework. We also discuss how to adapt SafeCompress to other attacks besides membership inference attack, demonstrating the flexibility of SafeCompress.

摘要: 人工智能(AI)软件中的深度学习模型的规模正在迅速增长，阻碍了在资源受限的设备(如智能手机)上的大规模部署。为了缓解这个问题，人工智能软件压缩起到了至关重要的作用，其目标是在保持高性能的同时压缩模型大小。然而，大模型中的固有缺陷可能会被压缩的模型继承。这样的缺陷很容易被攻击者利用，因为压缩模型通常部署在大量设备中，而没有足够的保护。在本文中，我们旨在从安全-性能联合优化的角度解决安全模型压缩问题。具体地说，受软件工程中测试驱动开发(TDD)范式的启发，我们提出了一个称为SafeCompress的测试驱动稀疏训练框架。通过将攻击机制模拟为安全测试，SafeCompress可以按照动态稀疏训练范式自动将大模型压缩为小模型。然后，考虑到两种典型的异构性攻击机制，即黑盒成员关系推理攻击和白盒成员关系推理攻击，我们开发了两个具体的实例：BMIA-SafeCompress和WMIA-SafeCompress。此外，我们实现了另一个实例MMIA-SafeCompress，通过扩展SafeCompress来防御对手同时进行黑盒和白盒成员推理攻击的情况。我们在计算机视觉和自然语言处理任务的五个数据集上进行了广泛的实验。结果表明，该框架具有较好的通用性和有效性。我们还讨论了如何使SafeCompress适应除成员推理攻击之外的其他攻击，展示了SafeCompress的灵活性。



## **47. Detection and Defense Against Prominent Attacks on Preconditioned LLM-Integrated Virtual Assistants**

对预置LLM集成虚拟助理的显著攻击的检测和防御 cs.CR

Accepted to be published in the Proceedings of the 10th IEEE CSDE  2023, the Asia-Pacific Conference on Computer Science and Data Engineering  2023

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00994v1) [paper-pdf](http://arxiv.org/pdf/2401.00994v1)

**Authors**: Chun Fai Chan, Daniel Wankit Yip, Aysan Esmradi

**Abstract**: The emergence of LLM (Large Language Model) integrated virtual assistants has brought about a rapid transformation in communication dynamics. During virtual assistant development, some developers prefer to leverage the system message, also known as an initial prompt or custom prompt, for preconditioning purposes. However, it is important to recognize that an excessive reliance on this functionality raises the risk of manipulation by malicious actors who can exploit it with carefully crafted prompts. Such malicious manipulation poses a significant threat, potentially compromising the accuracy and reliability of the virtual assistant's responses. Consequently, safeguarding the virtual assistants with detection and defense mechanisms becomes of paramount importance to ensure their safety and integrity. In this study, we explored three detection and defense mechanisms aimed at countering attacks that target the system message. These mechanisms include inserting a reference key, utilizing an LLM evaluator, and implementing a Self-Reminder. To showcase the efficacy of these mechanisms, they were tested against prominent attack techniques. Our findings demonstrate that the investigated mechanisms are capable of accurately identifying and counteracting the attacks. The effectiveness of these mechanisms underscores their potential in safeguarding the integrity and reliability of virtual assistants, reinforcing the importance of their implementation in real-world scenarios. By prioritizing the security of virtual assistants, organizations can maintain user trust, preserve the integrity of the application, and uphold the high standards expected in this era of transformative technologies.

摘要: LLM(Large Language Model，大型语言模型)集成虚拟助手的出现，带来了交流动力学的快速变革。在虚拟助手开发期间，一些开发人员更喜欢利用系统消息(也称为初始提示或自定义提示)进行预条件处理。但是，重要的是要认识到，过度依赖此功能会增加恶意攻击者操纵该功能的风险，恶意攻击者可以通过精心设计的提示来利用该功能。这种恶意操作构成了重大威胁，可能会损害虚拟助理响应的准确性和可靠性。因此，使用检测和防御机制来保护虚拟助理，对于确保其安全性和完整性至关重要。在本研究中，我们探索了三种检测和防御机制，旨在对抗以系统消息为目标的攻击。这些机制包括插入引用关键字、使用LLM求值器以及实现自我提醒。为了展示这些机制的有效性，他们针对突出的攻击技术进行了测试。我们的研究结果表明，所研究的机制能够准确地识别和对抗攻击。这些机制的有效性突出了它们在保障虚拟助理的完整性和可靠性方面的潜力，加强了在现实世界情景中执行这些机制的重要性。通过优先考虑虚拟助理的安全性，组织可以维护用户信任、维护应用程序的完整性，并保持在这个变革性技术时代所期望的高标准。



## **48. A Novel Evaluation Framework for Assessing Resilience Against Prompt Injection Attacks in Large Language Models**

一种新的评估大型语言模型抗即时注入攻击能力的评估框架 cs.CR

Accepted to be published in the Proceedings of The 10th IEEE CSDE  2023, the Asia-Pacific Conference on Computer Science and Data Engineering  2023

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00991v1) [paper-pdf](http://arxiv.org/pdf/2401.00991v1)

**Authors**: Daniel Wankit Yip, Aysan Esmradi, Chun Fai Chan

**Abstract**: Prompt injection attacks exploit vulnerabilities in large language models (LLMs) to manipulate the model into unintended actions or generate malicious content. As LLM integrated applications gain wider adoption, they face growing susceptibility to such attacks. This study introduces a novel evaluation framework for quantifying the resilience of applications. The framework incorporates innovative techniques designed to ensure representativeness, interpretability, and robustness. To ensure the representativeness of simulated attacks on the application, a meticulous selection process was employed, resulting in 115 carefully chosen attacks based on coverage and relevance. For enhanced interpretability, a second LLM was utilized to evaluate the responses generated from these simulated attacks. Unlike conventional malicious content classifiers that provide only a confidence score, the LLM-based evaluation produces a score accompanied by an explanation, thereby enhancing interpretability. Subsequently, a resilience score is computed by assigning higher weights to attacks with greater impact, thus providing a robust measurement of the application resilience. To assess the framework's efficacy, it was applied on two LLMs, namely Llama2 and ChatGLM. Results revealed that Llama2, the newer model exhibited higher resilience compared to ChatGLM. This finding substantiates the effectiveness of the framework, aligning with the prevailing notion that newer models tend to possess greater resilience. Moreover, the framework exhibited exceptional versatility, requiring only minimal adjustments to accommodate emerging attack techniques and classifications, thereby establishing itself as an effective and practical solution. Overall, the framework offers valuable insights that empower organizations to make well-informed decisions to fortify their applications against potential threats from prompt injection.

摘要: 提示注入攻击利用大型语言模型(LLM)中的漏洞将模型操纵为意外操作或生成恶意内容。随着LLM集成应用程序得到更广泛的采用，它们面临着越来越容易受到此类攻击的风险。这项研究介绍了一种新的评估框架，用于量化应用程序的弹性。该框架结合了旨在确保代表性、可解释性和健壮性的创新技术。为了确保对应用程序的模拟攻击的代表性，采用了精心选择的过程，根据覆盖范围和相关性精心选择了115个攻击。为了增强可解释性，使用了第二个LLM来评估这些模拟攻击生成的响应。与仅提供置信度分数的传统恶意内容分类器不同，基于LLM的评估会生成伴随解释的分数，从而增强可解释性。随后，通过将更高的权重分配给影响更大的攻击来计算弹性分数，从而提供对应用程序弹性的稳健测量。为了评估该框架的有效性，将其应用于两个LLM上，即Llama2和ChatGLM。结果表明，与ChatGLM相比，较新的模型Llama2表现出更高的弹性。这一发现证实了该框架的有效性，与流行的观点一致，即较新的模型往往具有更强的弹性。此外，该框架显示出非凡的多功能性，只需进行最小的调整即可适应新出现的攻击技术和分类，从而确立其本身是一种有效和实用的解决方案。总体而言，该框架提供了有价值的见解，使组织能够做出明智的决策，以加强其应用程序免受即时注入的潜在威胁。



## **49. Opening A Pandora's Box: Things You Should Know in the Era of Custom GPTs**

打开潘多拉的盒子：定制GPT时代你应该知道的事情 cs.CR

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2401.00905v1) [paper-pdf](http://arxiv.org/pdf/2401.00905v1)

**Authors**: Guanhong Tao, Siyuan Cheng, Zhuo Zhang, Junmin Zhu, Guangyu Shen, Xiangyu Zhang

**Abstract**: The emergence of large language models (LLMs) has significantly accelerated the development of a wide range of applications across various fields. There is a growing trend in the construction of specialized platforms based on LLMs, such as the newly introduced custom GPTs by OpenAI. While custom GPTs provide various functionalities like web browsing and code execution, they also introduce significant security threats. In this paper, we conduct a comprehensive analysis of the security and privacy issues arising from the custom GPT platform. Our systematic examination categorizes potential attack scenarios into three threat models based on the role of the malicious actor, and identifies critical data exchange channels in custom GPTs. Utilizing the STRIDE threat modeling framework, we identify 26 potential attack vectors, with 19 being partially or fully validated in real-world settings. Our findings emphasize the urgent need for robust security and privacy measures in the custom GPT ecosystem, especially in light of the forthcoming launch of the official GPT store by OpenAI.

摘要: 大型语言模型(LLM)的出现极大地加速了各个领域广泛应用的开发。基于LLMS的专业平台建设有日益增长的趋势，例如OpenAI新推出的定制GPT。虽然自定义GPT提供了各种功能，如Web浏览和代码执行，但它们也带来了重大的安全威胁。在本文中，我们对定制GPT平台产生的安全和隐私问题进行了全面分析。我们的系统检查根据恶意行为者的角色将潜在的攻击场景分类为三种威胁模型，并确定了自定义GPT中的关键数据交换通道。利用STRIDE威胁建模框架，我们识别了26个潜在的攻击向量，其中19个在现实世界中得到了部分或完全的验证。我们的发现强调了定制GPT生态系统中强大的安全和隐私措施的迫切需要，特别是考虑到OpenAI即将推出官方GPT商店。



## **50. Identifying and Mitigating the Security Risks of Generative AI**

识别和缓解生成性人工智能的安全风险 cs.AI

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2308.14840v4) [paper-pdf](http://arxiv.org/pdf/2308.14840v4)

**Authors**: Clark Barrett, Brad Boyd, Elie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.

摘要: 每一项重大技术发明都会重新面临两难境地--新技术既有可能被用来做好事，也有可能被用来做坏事。生成性人工智能(GenAI)技术，如大型语言模型(LLMS)和扩散模型，已经显示出非凡的能力(例如，上下文学习、代码完成以及文本到图像的生成和编辑)。然而，攻击者也可以利用GenAI来生成新的攻击，并提高现有攻击的速度和效率。本文报告了在谷歌(由斯坦福大学和威斯康星大学麦迪逊分校联合举办)举行的关于GenAI造成的两用困境的研讨会的结果。这篇论文并不是要全面的，而是试图综合研讨会的一些有趣的发现。我们就这一主题讨论社区的短期和长期目标。我们希望这篇论文既为讨论这一重要主题提供了一个起点，也为研究界可以努力解决的有趣问题提供了一个起点。



