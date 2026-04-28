#ifndef RIME_WANXIANG_FILTER_H_
#define RIME_WANXIANG_FILTER_H_

#include <rime/filter.h>
#include <rime/component.h>
#include <ncnn/net.h>
#include <map>
#include <string>
#include <vector>
#include <memory>

namespace rime {

class WanxiangFilter : public Filter {
 public:
  explicit WanxiangFilter(const Ticket& ticket);

  virtual an<Translation> Apply(an<Translation> translation,
                                CandidateList* candidates) override;

  float ScoreCandidate(const std::string& text, std::string& id_trace);
  
  std::vector<int> EncodeText(const std::string& text);
  size_t Utf8CharCount(const std::string& text);

 private:
  std::unique_ptr<ncnn::Net> net_;
  std::map<std::string, int> vocab_;
  bool model_loaded_ = false;
};

}  // namespace rime

#endif  // RIME_WANXIANG_FILTER_H_