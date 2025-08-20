/**
 * 师范生创意写作AI辅助系统 - 常量定义
 * 包含系统中使用的所有常量配置
 */

// ============ API配置常量 ============

/** API基础配置 */
export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080',
  TIMEOUT: 30000, // 30秒超时
  RETRY_TIMES: 3,
  RETRY_DELAY: 1000, // 1秒重试延迟
} as const;

/** API端点路径 */
export const API_ENDPOINTS = {
  // 认证相关
  AUTH: {
    LOGIN: '/api/v1/auth/login',
    LOGOUT: '/api/v1/auth/logout', 
    REFRESH: '/api/v1/auth/refresh',
    REGISTER: '/api/v1/auth/register',
  },
  
  // 用户相关
  USER: {
    PROFILE: '/api/v1/users/profile',
    PREFERENCES: '/api/v1/users/preferences',
    SESSIONS: '/api/v1/users/sessions',
    HISTORY: '/api/v1/users/history',
  },
  
  // 行为数据相关
  BEHAVIOR: {
    SUBMIT: '/api/v1/behavior-data',
    BATCH_SUBMIT: '/api/v1/behavior-data/batch',
    HISTORY: '/api/v1/behavior-data/history',
  },
  
  // AI推理相关
  INFERENCE: {
    PROCESS: '/api/v1/inference/process',
    RESULT: '/api/v1/inference/result',
    BATCH: '/api/v1/inference/batch',
  },
  
  // 认知状态相关
  COGNITIVE: {
    LATEST: '/api/v1/cognitive/latest',
    HISTORY: '/api/v1/cognitive/history',
    ANALYSIS: '/api/v1/cognitive/analysis',
  },
  
  // 任务相关
  TASKS: {
    LIST: '/api/v1/tasks',
    DETAIL: '/api/v1/tasks/:id',
    START: '/api/v1/tasks/:id/start',
    SUBMIT: '/api/v1/tasks/:id/submit',
    SESSIONS: '/api/v1/tasks/sessions',
  },
  
  // 数据导出
  EXPORT: {
    BEHAVIOR: '/api/v1/export/behavior',
    COGNITIVE: '/api/v1/export/cognitive',
    SESSIONS: '/api/v1/export/sessions',
    REPORT: '/api/v1/export/report',
  },
} as const;

// ============ WebSocket配置 ============

/** WebSocket配置 */
export const WEBSOCKET_CONFIG = {
  URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8080',
  RECONNECT_INTERVAL: 5000, // 5秒重连间隔
  MAX_RECONNECT_ATTEMPTS: 10,
  HEARTBEAT_INTERVAL: 30000, // 30秒心跳间隔
} as const;

/** WebSocket消息类型 */
export const WS_MESSAGE_TYPES = {
  // 客户端发送
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  HEARTBEAT: 'heartbeat',
  JOIN_ROOM: 'join_room',
  LEAVE_ROOM: 'leave_room',
  BEHAVIOR_DATA: 'behavior_data',
  INFERENCE_REQUEST: 'inference_request',
  
  // 服务端推送
  CONNECTION_ACK: 'connection_ack',
  COGNITIVE_UPDATE: 'cognitive_update',
  INFERENCE_RESULT: 'inference_result',
  SYSTEM_NOTIFICATION: 'system_notification',
  ERROR: 'error',
} as const;

// ============ 认知概念配置 ============

/** 认知概念定义 */
export const COGNITIVE_CONCEPTS = {
  attention: {
    name: '注意力',
    description: '集中注意力和专注能力',
    icon: 'eye',
    color: '#1890ff',
    range: [0, 1],
    levels: {
      low: { min: 0, max: 0.3, label: '较低', color: '#ff4d4f' },
      medium: { min: 0.3, max: 0.7, label: '中等', color: '#faad14' },
      high: { min: 0.7, max: 1, label: '较高', color: '#52c41a' },
    },
  },
  memory: {
    name: '工作记忆',
    description: '信息存储和处理能力',
    icon: 'database',
    color: '#722ed1',
    range: [0, 1],
    levels: {
      low: { min: 0, max: 0.3, label: '较低', color: '#ff4d4f' },
      medium: { min: 0.3, max: 0.7, label: '中等', color: '#faad14' },
      high: { min: 0.7, max: 1, label: '较高', color: '#52c41a' },
    },
  },
  comprehension: {
    name: '理解力',
    description: '理解和分析能力', 
    icon: 'bulb',
    color: '#13c2c2',
    range: [0, 1],
    levels: {
      low: { min: 0, max: 0.3, label: '较低', color: '#ff4d4f' },
      medium: { min: 0.3, max: 0.7, label: '中等', color: '#faad14' },
      high: { min: 0.7, max: 1, label: '较高', color: '#52c41a' },
    },
  },
  creativity: {
    name: '创造力',
    description: '创新思维和想象力',
    icon: 'rocket',
    color: '#eb2f96',
    range: [0, 1],
    levels: {
      low: { min: 0, max: 0.3, label: '较低', color: '#ff4d4f' },
      medium: { min: 0.3, max: 0.7, label: '中等', color: '#faad14' },
      high: { min: 0.7, max: 1, label: '较高', color: '#52c41a' },
    },
  },
  motivation: {
    name: '动机水平',
    description: '学习动机和积极性',
    icon: 'fire',
    color: '#fa8c16',
    range: [0, 1],
    levels: {
      low: { min: 0, max: 0.3, label: '较低', color: '#ff4d4f' },
      medium: { min: 0.3, max: 0.7, label: '中等', color: '#faad14' },
      high: { min: 0.7, max: 1, label: '较高', color: '#52c41a' },
    },
  },
  emotion: {
    name: '情绪状态',
    description: '情绪稳定性和积极程度',
    icon: 'heart',
    color: '#f759ab',
    range: [0, 1],
    levels: {
      low: { min: 0, max: 0.3, label: '消极', color: '#ff4d4f' },
      medium: { min: 0.3, max: 0.7, label: '平稳', color: '#faad14' },
      high: { min: 0.7, max: 1, label: '积极', color: '#52c41a' },
    },
  },
  confidence: {
    name: '自信心',
    description: '对自己能力的信心',
    icon: 'trophy',
    color: '#a0d911',
    range: [0, 1],
    levels: {
      low: { min: 0, max: 0.3, label: '较低', color: '#ff4d4f' },
      medium: { min: 0.3, max: 0.7, label: '中等', color: '#faad14' },
      high: { min: 0.7, max: 1, label: '较高', color: '#52c41a' },
    },
  },
  fatigue: {
    name: '疲劳度',
    description: '身心疲劳程度',
    icon: 'clock-circle',
    color: '#bfbfbf',
    range: [0, 1],
    levels: {
      low: { min: 0, max: 0.3, label: '精力充沛', color: '#52c41a' },
      medium: { min: 0.3, max: 0.7, label: '轻微疲劳', color: '#faad14' },
      high: { min: 0.7, max: 1, label: '明显疲劳', color: '#ff4d4f' },
    },
  },
} as const;

// ============ 数据采集配置 ============

/** 行为数据采集配置 */
export const BEHAVIOR_COLLECTION = {
  // 采集间隔 (毫秒)
  INTERVALS: {
    MOUSE_SAMPLING: 100, // 鼠标采样间隔
    KEYBOARD_SAMPLING: 50, // 键盘采样间隔
    STATE_UPDATE: 5000, // 认知状态更新间隔
    DATA_UPLOAD: 10000, // 数据上传间隔
  },
  
  // 缓冲区大小
  BUFFER_SIZES: {
    MOUSE_EVENTS: 1000,
    KEYBOARD_EVENTS: 500,
    TEXT_CHANGES: 200,
  },
  
  // 过滤器配置
  FILTERS: {
    MIN_MOUSE_MOVEMENT: 5, // 最小鼠标移动距离 (像素)
    MAX_IDLE_TIME: 60000, // 最大空闲时间 (毫秒)
    EXCLUDE_KEYS: ['Meta', 'Alt', 'Control', 'Shift'], // 排除的按键
  },
} as const;

// ============ UI界面配置 ============

/** 界面主题配置 */
export const THEME_CONFIG = {
  COLORS: {
    PRIMARY: '#1890ff',
    SUCCESS: '#52c41a',
    WARNING: '#faad14',
    ERROR: '#ff4d4f',
    INFO: '#13c2c2',
    TEXT_PRIMARY: '#262626',
    TEXT_SECONDARY: '#595959',
    BACKGROUND: '#fafafa',
    BORDER: '#d9d9d9',
  },
  
  LAYOUT: {
    HEADER_HEIGHT: 64,
    SIDEBAR_WIDTH: 256,
    SIDEBAR_COLLAPSED_WIDTH: 80,
    FOOTER_HEIGHT: 48,
  },
  
  ANIMATION: {
    DURATION_FAST: 0.2,
    DURATION_NORMAL: 0.3,
    DURATION_SLOW: 0.5,
    EASING: 'cubic-bezier(0.645, 0.045, 0.355, 1)',
  },
} as const;

/** 响应式断点 */
export const BREAKPOINTS = {
  xs: 480,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1600,
} as const;

// ============ 任务配置 ============

/** 任务类型配置 */
export const TASK_TYPES = {
  writing: {
    name: '创意写作',
    description: '进行创意写作练习',
    icon: 'edit',
    color: '#1890ff',
    estimatedTime: 30, // 预估时间(分钟)
  },
  comprehension: {
    name: '阅读理解',
    description: '阅读理解训练',
    icon: 'read',
    color: '#52c41a',
    estimatedTime: 20,
  },
  creativity: {
    name: '创造力训练',
    description: '创造性思维训练',
    icon: 'rocket',
    color: '#eb2f96',
    estimatedTime: 25,
  },
} as const;

/** 任务难度配置 */
export const TASK_DIFFICULTIES = {
  easy: {
    name: '简单',
    color: '#52c41a',
    multiplier: 0.8,
  },
  medium: {
    name: '中等',
    color: '#faad14',
    multiplier: 1.0,
  },
  hard: {
    name: '困难',
    color: '#ff4d4f',
    multiplier: 1.3,
  },
} as const;

// ============ 数据可视化配置 ============

/** 图表颜色配置 */
export const CHART_COLORS = [
  '#1890ff', '#52c41a', '#faad14', '#ff4d4f',
  '#722ed1', '#13c2c2', '#eb2f96', '#fa8c16',
  '#a0d911', '#2f54eb', '#f759ab', '#bfbfbf',
] as const;

/** 图表默认配置 */
export const CHART_CONFIG = {
  GRID: {
    top: 60,
    right: 40,
    bottom: 60,
    left: 60,
  },
  
  ANIMATION: {
    duration: 1000,
    easing: 'cubicOut',
  },
  
  TOOLTIP: {
    trigger: 'axis',
    backgroundColor: 'rgba(50, 50, 50, 0.8)',
    borderColor: 'transparent',
    textStyle: {
      color: '#fff',
    },
  },
} as const;

// ============ 错误码定义 ============

/** 错误码常量 */
export const ERROR_CODES = {
  // 网络错误
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT_ERROR: 'TIMEOUT_ERROR',
  
  // 认证错误
  UNAUTHORIZED: 'UNAUTHORIZED',
  TOKEN_EXPIRED: 'TOKEN_EXPIRED',
  FORBIDDEN: 'FORBIDDEN',
  
  // 数据错误
  INVALID_DATA: 'INVALID_DATA',
  DATA_NOT_FOUND: 'DATA_NOT_FOUND',
  DUPLICATE_DATA: 'DUPLICATE_DATA',
  
  // 系统错误
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',
  
  // 业务错误
  INFERENCE_FAILED: 'INFERENCE_FAILED',
  BEHAVIOR_COLLECTION_FAILED: 'BEHAVIOR_COLLECTION_FAILED',
  COGNITIVE_STATE_ERROR: 'COGNITIVE_STATE_ERROR',
} as const;

/** 错误信息映射 */
export const ERROR_MESSAGES = {
  [ERROR_CODES.NETWORK_ERROR]: '网络连接失败，请检查网络设置',
  [ERROR_CODES.TIMEOUT_ERROR]: '请求超时，请稍后重试',
  [ERROR_CODES.UNAUTHORIZED]: '未授权访问，请重新登录',
  [ERROR_CODES.TOKEN_EXPIRED]: '登录已过期，请重新登录',
  [ERROR_CODES.FORBIDDEN]: '权限不足，无法访问',
  [ERROR_CODES.INVALID_DATA]: '数据格式错误',
  [ERROR_CODES.DATA_NOT_FOUND]: '数据不存在',
  [ERROR_CODES.DUPLICATE_DATA]: '数据已存在',
  [ERROR_CODES.INTERNAL_ERROR]: '系统内部错误',
  [ERROR_CODES.SERVICE_UNAVAILABLE]: '服务暂时不可用',
  [ERROR_CODES.INFERENCE_FAILED]: 'AI推理失败',
  [ERROR_CODES.BEHAVIOR_COLLECTION_FAILED]: '行为数据采集失败',
  [ERROR_CODES.COGNITIVE_STATE_ERROR]: '认知状态计算错误',
} as const;

// ============ 本地存储Key ============

/** 本地存储键名 */
export const STORAGE_KEYS = {
  // 认证相关
  ACCESS_TOKEN: 'creative_writing_access_token',
  REFRESH_TOKEN: 'creative_writing_refresh_token',
  USER_INFO: 'creative_writing_user_info',
  
  // 用户偏好
  USER_PREFERENCES: 'creative_writing_preferences',
  THEME: 'creative_writing_theme',
  LANGUAGE: 'creative_writing_language',
  
  // 应用状态
  SIDEBAR_COLLAPSED: 'creative_writing_sidebar_collapsed',
  LAST_VISITED_PAGE: 'creative_writing_last_page',
  
  // 临时数据
  DRAFT_CONTENT: 'creative_writing_draft',
  SESSION_DATA: 'creative_writing_session',
} as const;

// ============ 默认配置值 ============

/** 默认用户偏好设置 */
export const DEFAULT_PREFERENCES = {
  theme: 'light',
  language: 'zh-CN',
  notifications: {
    cognitive: true,
    performance: true,
    system: true,
  },
  privacy: {
    shareData: false,
    recordBehavior: true,
  },
} as const;

/** 默认分页配置 */
export const DEFAULT_PAGINATION = {
  page: 1,
  pageSize: 20,
  showSizeChanger: true,
  showQuickJumper: true,
  showTotal: (total: number) => `共 ${total} 条记录`,
} as const;