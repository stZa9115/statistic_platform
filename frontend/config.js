// === 全站後端 API 設定 ===
laptop = false;

// 開發 / 測試 / 臨時展示
window.API_CONFIG = {
  BASE_URL: laptop
    ? "http://127.0.0.1:5000"
    : "http://140.115.197.64:5581", //只有開給中央那台 154:~/ncubacktest/ncuback$有設定全開放的 main.py

  // 你也可以順便集中所有 endpoint
  ENDPOINTS: {
    independentTtest: "/ttest/independentTtest/upload",
    pairedTtest: "/ttest/pairedTtest/upload",
    anova: "/anova/anova/upload",
    download: "/stat/download",
    downloadZip: "/stat/download_zip"
  }
};

console.log("API url:", window.API_CONFIG.BASE_URL);