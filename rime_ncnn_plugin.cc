#include <rime/component.h>
#include <rime/filter.h>
#include <rime/translation.h>
#include <rime/candidate.h>
#include <rime/registry.h>
#include <net.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace rime;

// ==========================================
// 极速重排引擎：融合旁路拦截与动态剪枝
// ==========================================
class NcnnFilter : public Filter {
public:
    explicit NcnnFilter(const Ticket& ticket);
    virtual an<Translation> Apply(an<Translation> translation, CandidateList* candidates);

private:
    ncnn::Net net_;
    std::unordered_map<std::string, int> vocab_;
    bool model_loaded_ = false;

    // 辅助函数
    size_t Utf8CharCount(const std::string& text);
    std::vector<int> EncodeText(const std::string& text);
    float ScoreCandidate(const std::string& text);
};

// --- 1. 初始化与模型加载 ---
NcnnFilter::NcnnFilter(const Ticket& ticket) : Filter(ticket) {
    // 【修改此处为你桌面上 .param 和 .bin 的绝对路径】
    int param_ret = net_.load_param("/home/amz/Desktop/rime_morandi_lm.ncnn.param");
    int bin_ret = net_.load_model("/home/amz/Desktop/rime_morandi_lm.ncnn.bin");

    if (param_ret == 0 && bin_ret == 0) {
        model_loaded_ = true;
        LOG(INFO) << "[NCNN] Model loaded successfully! Ready for strike.";
    } else {
        LOG(ERROR) << "[NCNN] Failed to load model!";
    }

    // 初始化词表 (0 为 未知词 UNK)
    vocab_["公"] = 1; vocab_["众"] = 2; vocab_["号"] = 3; 
    vocab_["尤"] = 4; vocab_["其"] = 5; vocab_["有"] = 6;
    vocab_["特"] = 7; vocab_["别"] = 8;
    // ... 在此补全你的其他汉字映射 ...
}

// --- 2. 统计真实中文字数 ---
size_t NcnnFilter::Utf8CharCount(const std::string& text) {
    size_t count = 0;
    for (size_t i = 0; i < text.length(); ) {
        if ((text[i] & 0xf8) == 0xf0) i += 4;
        else if ((text[i] & 0xf0) == 0xe0) i += 3;
        else if ((text[i] & 0xe0) == 0xc0) i += 2;
        else i += 1;
        count++;
    }
    return count;
}

// --- 3. UTF-8 切字转 Tensor ID ---
std::vector<int> NcnnFilter::EncodeText(const std::string& text) {
    std::vector<int> ids;
    for (size_t i = 0; i < text.length(); ) {
        int cplen = 1;
        if ((text[i] & 0xf8) == 0xf0) cplen = 4;
        else if ((text[i] & 0xf0) == 0xe0) cplen = 3;
        else if ((text[i] & 0xe0) == 0xc0) cplen = 2;
        
        std::string char_str = text.substr(i, cplen);
        if (vocab_.count(char_str)) {
            ids.push_back(vocab_[char_str]);
        } else {
            ids.push_back(0); // 遇到没见过的字给 UNK
        }
        i += cplen;
    }
    return ids;
}

// --- 4. 神经网络矩阵推理打分 ---
float NcnnFilter::ScoreCandidate(const std::string& text) {
    if (!model_loaded_) return 0.0f;
    std::vector<int> ids = EncodeText(text);
    if (ids.size() < 2) return 0.0f;

    ncnn::Mat in(ids.size() - 1, 1);
    for (size_t i = 0; i < ids.size() - 1; i++) {
        in[i] = (float)ids[i];
    }

    ncnn::Extractor ex = net_.create_extractor();
    ex.input("input_ids", in);

    ncnn::Mat out;
    ex.extract("logits", out);

    float total_score = 0.0f;
    for (size_t i = 0; i < ids.size() - 1; i++) {
        int target_id = ids[i + 1];
        const float* out_ptr = out.row(i); 
        total_score += out_ptr[target_id];
    }
    return total_score;
}

// --- 5. 核心逻辑：拦截、重排与剪枝 ---
an<Translation> NcnnFilter::Apply(an<Translation> translation, CandidateList* candidates) {
    if (!translation) return nullptr;

    std::vector<an<Candidate>> cands;
    // 拦截 Rime 吐出的前 5 个候选词
    while (!translation->Exhausted() && cands.size() < 5) {
        cands.push_back(translation->Peek());
        translation->Next();
    }

    if (cands.empty()) return nullptr;

    // 结构体：用来暂存候选词的打分状态
    struct ScoredCand {
        an<Candidate> cand;
        float nn_score;
        bool is_bypassed;
    };

    std::vector<ScoredCand> scored_list;
    float max_nn_score = -1e9;

    // 遍历这 5 个候选词，进行【旁路拦截判定】
    for (auto& cand : cands) {
        std::string type = cand->type();
        size_t char_len = Utf8CharCount(cand->text());

        // 旁路规则：如果是固定词组(table/user_phrase)，或者字数太少，直接免检放行！
        if (type != "sentence" || char_len < 3) {
            scored_list.push_back({cand, 0.0f, true});
        } else {
            // 长句：交由神经网络进行降维打击
            float s = ScoreCandidate(cand->text());
            scored_list.push_back({cand, s, false});
            if (s > max_nn_score) max_nn_score = s;
        }
    }

    std::vector<an<Candidate>> final_cands;   // 最终展示的高优列表
    std::vector<an<Candidate>> pruned_cands;  // 被剪枝的劣质列表
    std::vector<ScoredCand> nn_cands;         // 参与排序的句子列表

    float margin_threshold = 8.0f; // 【边缘剪枝落差阈值】可自行微调

    // 执行【动态边缘剪枝】
    for (auto& sc : scored_list) {
        if (sc.is_bypassed) {
            // 固定词库的词享有最高特权，原封不动推入展示列表
            final_cands.push_back(sc.cand);
        } else {
            // 句子落后最高分太多，直接扔进垃圾桶（降权备用组）
            if (max_nn_score - sc.nn_score > margin_threshold) {
                pruned_cands.push_back(sc.cand);
            } else {
                nn_cands.push_back(sc);
            }
        }
    }

    // 对成功活下来的句子，按神经网络得分【降序重排】
    std::sort(nn_cands.begin(), nn_cands.end(), [](const ScoredCand& a, const ScoredCand& b) {
        return a.nn_score > b.nn_score;
    });

    // 拼装最终结果： 特权固定词 -> NN 高分优选句 
    for (auto& sc : nn_cands) {
        final_cands.push_back(sc.cand);
    }

    // 将结果写入新的迭代器
    auto reordered = New<vector<an<Candidate>>>();
    for (auto& c : final_cands) {
        reordered->push_back(c);
    }
    
    // 宽容机制：把落差太大的烂句子放在这 5 个词的最末尾（防止用户真的想打错别字）
    for (auto& c : pruned_cands) {
        reordered->push_back(c);
    }

    // 将翻译器里剩下的第 6、7、8...个词接在最后，保证不吞词
    while (!translation->Exhausted()) {
        reordered->push_back(translation->Peek());
        translation->Next();
    }

    return New<ListTranslation>(reordered);
}

// ==========================================
// 插件注册入口
// ==========================================
static void rime_ncnn_initialize() {
    LOG(INFO) << "Registering ncnn neural filter...";
    Registry& r = Registry::instance();
    // 挂载名：ncnn_filter
    r.Register("ncnn_filter", new Component<NcnnFilter>);
}

static void rime_ncnn_finalize() {}

RIME_REGISTER_MODULE(ncnn)