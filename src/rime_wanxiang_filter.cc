#include "rime_wanxiang_filter.h"
#include <ncnn/net.h>
#include <fstream>
#include <algorithm>
#include <vector>
#include <memory>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <sstream>
#include <clocale>
#include <sys/stat.h>
#include <rime/candidate.h>
#include <rime/translation.h>
#include <rime/service.h>
#include <rime/deployer.h>

namespace rime {

class WanxiangTranslation : public Translation {
 public:
  WanxiangTranslation(CandidateList cands) : cands_(std::move(cands)), cursor_(0) {}
  virtual bool exhausted() const { return cursor_ >= cands_.size(); }
  virtual an<Candidate> Peek() { return (cursor_ < cands_.size()) ? cands_[cursor_] : nullptr; }
  virtual bool Next() { if (cursor_ >= cands_.size()) return false; ++cursor_; return true; }
 private:
  CandidateList cands_;
  size_t cursor_;
};

std::string GetPreciseTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

WanxiangFilter::WanxiangFilter(const Ticket& ticket) : Filter(ticket) {
    std::setlocale(LC_ALL, "C");

    // 💡 核心改动 1：调用 Rime API 动态获取当前系统的 user_data_dir！
    // 彻底告别硬编码，实现全平台兼容！
    std::string user_dir = Service::instance().deployer().user_data_dir.string();
    std::string log_p = user_dir + "/wanxiang.log";
    
    auto L = [&](const std::string& tag, const std::string& msg) {
        std::ofstream f(log_p, std::ios::app);
        if (f.is_open()) f << "[" << GetPreciseTimestamp() << "] [" << tag << "] " << msg << std::endl;
        LOG(INFO) << "[Wanxiang] " << tag << ": " << msg;
    };

    L("INIT", "=== AI 核心启动 ===");
    L("INIT", "当前用户目录: " + user_dir);

    net_ = std::make_unique<ncnn::Net>();
    net_->opt.use_vulkan_compute = false;
    net_->opt.num_threads = 1;

    std::string param_path = user_dir + "/rime_morandi_lm.ncnn.param";
    std::string model_path = user_dir + "/rime_morandi_lm.ncnn.model";
    std::string vocab_path = user_dir + "/vocab.txt";

    // 词表加载
    std::ifstream v_f(vocab_path);
    if (v_f.is_open()) {
        std::string line; int id = 0;
        while (std::getline(v_f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) vocab_[line] = id;
            id++;
        }
        v_f.close();
    } else {
        L("INIT", "ERROR: 找不到词表文件 vocab.txt");
    }

    // 💡 核心改动 2：删除了危险的 stderr 劫持代码
    L("INIT", "正在解析 " + param_path);
    int p_ret = net_->load_param(param_path.c_str());
    
    if (p_ret == 0) {
        L("INIT", "正在加载 " + model_path);
        int b_ret = net_->load_model(model_path.c_str());
        if (b_ret == 0) {
            model_loaded_ = true;
            L("INIT", "🎉 SUCCESS: AI 核心已就绪！");
        } else {
            L("INIT", "ERROR: 权重文件加载失败 (错误码 " + std::to_string(b_ret) + ")");
        }
    } else {
        L("INIT", "ERROR: 网络结构解析失败 (错误码 " + std::to_string(p_ret) + ")");
    }
}

size_t WanxiangFilter::Utf8CharCount(const std::string& text) {
    size_t count = 0;
    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) i += 1; else if (c < 0xe0) i += 2; else if (c < 0xf0) i += 3; else i += 4;
        count++;
    }
    return count;
}

std::vector<int> WanxiangFilter::EncodeText(const std::string& text) {
    std::vector<int> ids;
    for (size_t i = 0; i < text.length(); ) {
        int cplen = 1;
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c >= 0xf0) cplen = 4; else if (c >= 0xe0) cplen = 3; else if (c >= 0xc0) cplen = 2;
        std::string char_str = text.substr(i, cplen);
        auto it = vocab_.find(char_str);
        ids.push_back(it != vocab_.end() ? it->second : 0);
        i += cplen;
    }
    return ids;
}

float WanxiangFilter::ScoreCandidate(const std::string& text, std::string& id_trace) {
    if (!model_loaded_) return -999.0f;
    std::vector<int> ids = EncodeText(text);
    if (ids.size() < 2) return -999.0f;
    std::stringstream ss; ss << "IDs: ["; for(int id : ids) ss << id << " "; ss << "]"; id_trace = ss.str();

    ncnn::Mat in(static_cast<int>(ids.size() - 1), 1);
    for (size_t i = 0; i < ids.size() - 1; i++) in[i] = static_cast<float>(ids[i]);

    ncnn::Extractor ex = net_->create_extractor();
    ex.input("in0", in); 
    ncnn::Mat out; ex.extract("out0", out);

    float total_s = 0.0f; int count = 0;
    for (size_t i = 0; i < ids.size() - 1; i++) {
        int tid = ids[i + 1];
        if (tid >= out.w || tid <= 0) continue;
        total_s += out.row(i)[tid]; count++;
    }
    return count > 0 ? (total_s / (float)count) : -999.0f;
}

an<Translation> WanxiangFilter::Apply(an<Translation> translation, CandidateList* candidates) {
    if (!translation || !model_loaded_) return translation;
    
    // 💡 核心改动 3：Apply 运行时的日志路径也改为动态获取
    std::string log_p = Service::instance().deployer().user_data_dir.string() + "/wanxiang.log";
    std::ofstream f(log_p, std::ios::app);
    auto L = [&](const std::string& m) { if (f.is_open()) f << "[" << GetPreciseTimestamp() << "] [RUN] " << m << std::endl; };

    CandidateList top;
    while (!translation->exhausted() && top.size() < 10) { top.push_back(translation->Peek()); translation->Next(); }
    if (top.empty()) return nullptr;

    struct ScoredItem { an<Candidate> cand; float s; bool is_ai; };
    std::vector<ScoredItem> analysis;
    float max_s = -1e9f;

    for (auto& cand : top) {
        if (cand->type() == "sentence" && Utf8CharCount(cand->text()) >= 3) {
            std::string trace; float s = ScoreCandidate(cand->text(), trace);
            L("AI: [" + cand->text() + "] = " + std::to_string(s) + " | " + trace);
            analysis.push_back({cand, s, true});
            if (s > max_s) max_s = s;
        } else { analysis.push_back({cand, 0.0f, false}); }
    }

    CandidateList res;
    std::vector<ScoredItem> survivors;
    for (auto& it : analysis) if (it.is_ai && max_s - it.s <= 15.0f) survivors.push_back(it);
    std::sort(survivors.begin(), survivors.end(), [](const ScoredItem& a, const ScoredItem& b){ return a.s > b.s; });
    for (auto& s : survivors) res.push_back(s.cand);
    for (auto& it : analysis) if (!it.is_ai) res.push_back(it.cand);
    for (auto& it : analysis) if (it.is_ai && (max_s - it.s > 15.0f)) res.push_back(it.cand);
    while (!translation->exhausted()) { res.push_back(translation->Peek()); translation->Next(); }
    f.close();
    return New<WanxiangTranslation>(std::move(res));
}

} // namespace rime