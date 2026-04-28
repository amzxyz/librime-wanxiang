#include <rime/common.h>
#include <rime/component.h>
#include <rime/registry.h>
#include <rime_api.h>
#include "rime_wanxiang_filter.h"

// 💡 定义初始化函数
static void rime_wanxiang_initialize() {
  rime::Registry& r = rime::Registry::instance();
  // 注册组件名，这个名字就是你在 default.custom.yaml 里的 filters 列表中填写的名字
  r.Register("wanxiang_filter", new rime::Component<rime::WanxiangFilter>);
}

static void rime_wanxiang_finalize() {
  // 退出时清理（通常为空）
}

// 💡 使用官方标准宏定义模块
// 第一个参数是模块名（对应文件名 librime-wanxiang.so）
// 第二个参数是初始化函数，第三个是清理函数
RIME_REGISTER_MODULE(wanxiang)

// 💡 补全分组挂载规范（这决定了 Rime 核心能否正确识别该插件组）
RIME_REGISTER_MODULE_GROUP(wanxiang, "wanxiang")