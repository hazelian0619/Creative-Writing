/**
 * 师范生创意写作AI辅助系统 - TypeScript类型定义
 * 定义系统中使用的所有数据类型和接口
 */

// ============ 基础类型定义 ============

/** 用户基础信息 */
export interface User {
  id: string;
  username: string;
  email: string;
  role: 'student' | 'teacher' | 'admin';
  profile: UserProfile;
  createdAt: string;
  lastLoginAt?: string;
}

/** 用户个人资料 */
export interface UserProfile {
  displayName: string;
  avatar?: string;
  grade?: string;
  school?: string;
  major?: string;
}

// ============ 认知状态相关类型 ============

/** 认知概念枚举 */
export type CognitiveConcept = 
  | 'attention'      // 注意力
  | 'memory'         // 工作记忆
  | 'comprehension'  // 理解力
  | 'creativity'     // 创造力
  | 'motivation'     // 动机水平
  | 'emotion'        // 情绪状态
  | 'confidence'     // 自信心
  | 'fatigue';       // 疲劳度

/** 认知状态向量 */
export interface CognitiveState {
  userId: string;
  sessionId: string;
  timestamp: number;
  values: Record<CognitiveConcept, number>; // 0-1之间的值
  confidence: number; // 预测置信度
  metadata?: {
    modelVersion: string;
    processingTime: number;
    rawFeatures?: any;
  };
}

/** 认知状态历史记录 */
export interface CognitiveStateHistory {
  states: CognitiveState[];
  timeRange: {
    start: number;
    end: number;
  };
  statistics: {
    averageValues: Record<CognitiveConcept, number>;
    trends: Record<CognitiveConcept, 'increasing' | 'decreasing' | 'stable'>;
  };
}

// ============ 用户行为数据类型 ============

/** 键盘事件数据 */
export interface KeystrokeEvent {
  key: string;
  timestamp: number;
  type: 'keydown' | 'keyup' | 'keypress';
  duration?: number; // 按键持续时间
  interval?: number; // 与上次按键的间隔
}

/** 鼠标事件数据 */
export interface MouseEvent {
  x: number;
  y: number;
  timestamp: number;
  type: 'mousemove' | 'click' | 'scroll' | 'drag';
  button?: 'left' | 'right' | 'middle';
  velocity?: number; // 移动速度
  acceleration?: number; // 加速度
}

/** 用户行为数据包 */
export interface UserBehaviorData {
  userId: string;
  sessionId: string;
  timestamp: number;
  taskContext: {
    taskId: string;
    taskType: 'writing' | 'reading' | 'thinking';
    prompt?: string;
  };
  keystrokes: KeystrokeEvent[];
  mouseData: MouseEvent[];
  textInput?: {
    content: string;
    wordCount: number;
    editCount: number;
  };
  duration: number; // 总时长（毫秒）
}

// ============ AI推理相关类型 ============

/** AI推理请求 */
export interface InferenceRequest {
  requestId: string;
  userId: string;
  inputData: {
    text: string;
    timestamp: number;
    context?: any;
  };
  contextHistory: UserBehaviorData[];
  options?: {
    includeExplanation: boolean;
    responseFormat: 'text' | 'structured';
  };
}

/** AI推理响应 */
export interface InferenceResponse {
  requestId: string;
  success: boolean;
  result?: {
    cognitiveState: CognitiveState;
    promptSuggestions: PromptSuggestion[];
    explanation?: string;
    confidence: number;
  };
  error?: string;
  processingTime: number;
}

/** Prompt建议 */
export interface PromptSuggestion {
  id: string;
  type: 'inspiration' | 'technique' | 'structure' | 'style';
  content: string;
  description: string;
  targetConcepts: CognitiveConcept[];
  difficulty: 'easy' | 'medium' | 'hard';
  estimatedTime: number; // 预估使用时间（分钟）
}

// ============ 实验和评估类型 ============

/** 实验任务 */
export interface ExperimentTask {
  id: string;
  title: string;
  description: string;
  type: 'writing' | 'comprehension' | 'creativity';
  prompt: string;
  timeLimit?: number; // 时间限制（分钟）
  targetConcepts: CognitiveConcept[];
  evaluationCriteria: string[];
}

/** 用户任务会话 */
export interface TaskSession {
  id: string;
  userId: string;
  taskId: string;
  startTime: number;
  endTime?: number;
  status: 'active' | 'completed' | 'aborted';
  behaviorData: UserBehaviorData[];
  cognitiveStates: CognitiveState[];
  submissions: TaskSubmission[];
  performance?: PerformanceMetrics;
}

/** 任务提交 */
export interface TaskSubmission {
  id: string;
  sessionId: string;
  content: string;
  timestamp: number;
  wordCount: number;
  editHistory: EditEvent[];
  finalCognitiveState: CognitiveState;
}

/** 编辑事件 */
export interface EditEvent {
  timestamp: number;
  type: 'insert' | 'delete' | 'modify';
  position: number;
  content: string;
  length: number;
}

/** 性能指标 */
export interface PerformanceMetrics {
  writingQuality: number; // 写作质量评分 (0-100)
  creativityScore: number; // 创造力评分 (0-100)
  taskCompletion: number; // 任务完成度 (0-100)
  timeEfficiency: number; // 时间效率 (0-100)
  cognitiveStability: number; // 认知状态稳定性 (0-100)
  improvementRate: number; // 提升率 (百分比)
}

// ============ 数据可视化类型 ============

/** 图表配置 */
export interface ChartConfig {
  type: 'line' | 'bar' | 'radar' | 'heatmap' | 'scatter';
  title: string;
  xAxis: AxisConfig;
  yAxis: AxisConfig;
  series: SeriesConfig[];
  colors?: string[];
  animation?: boolean;
}

/** 坐标轴配置 */
export interface AxisConfig {
  label: string;
  type: 'category' | 'value' | 'time';
  min?: number;
  max?: number;
  format?: string;
}

/** 数据系列配置 */
export interface SeriesConfig {
  name: string;
  data: any[];
  type?: string;
  smooth?: boolean;
  areaStyle?: any;
}

// ============ WebSocket通信类型 ============

/** WebSocket消息基础接口 */
export interface WebSocketMessage {
  type: string;
  timestamp: number;
  data: any;
}

/** 实时认知状态更新消息 */
export interface CognitiveStateUpdateMessage extends WebSocketMessage {
  type: 'cognitive_state_update';
  data: {
    userId: string;
    sessionId: string;
    cognitiveState: CognitiveState;
  };
}

/** AI推理结果消息 */
export interface InferenceResultMessage extends WebSocketMessage {
  type: 'inference_result';
  data: InferenceResponse;
}

/** 系统通知消息 */
export interface SystemNotificationMessage extends WebSocketMessage {
  type: 'system_notification';
  data: {
    level: 'info' | 'warning' | 'error' | 'success';
    title: string;
    content: string;
    duration?: number;
  };
}

// ============ API响应类型 ============

/** 标准API响应格式 */
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  timestamp: number;
  requestId?: string;
}

/** 分页响应 */
export interface PaginatedResponse<T = any> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
}

// ============ Redux Store类型 ============

/** 根状态类型 */
export interface RootState {
  auth: AuthState;
  user: UserState;
  cognitive: CognitiveState;
  task: TaskState;
  ui: UIState;
}

/** 认证状态 */
export interface AuthState {
  isAuthenticated: boolean;
  token?: string;
  user?: User;
  loading: boolean;
  error?: string;
}

/** 用户状态 */
export interface UserState {
  profile?: UserProfile;
  preferences: UserPreferences;
  sessions: TaskSession[];
  loading: boolean;
}

/** 用户偏好设置 */
export interface UserPreferences {
  theme: 'light' | 'dark';
  language: 'zh-CN' | 'en-US';
  notifications: {
    cognitive: boolean;
    performance: boolean;
    system: boolean;
  };
  privacy: {
    shareData: boolean;
    recordBehavior: boolean;
  };
}

/** 任务状态 */
export interface TaskState {
  currentTask?: ExperimentTask;
  currentSession?: TaskSession;
  history: TaskSession[];
  loading: boolean;
  error?: string;
}

/** UI状态 */
export interface UIState {
  sidebarCollapsed: boolean;
  theme: 'light' | 'dark';
  loading: {
    global: boolean;
    cognitive: boolean;
    inference: boolean;
  };
  notifications: SystemNotificationMessage[];
}

// ============ 组件Props类型 ============

/** 通用组件Props */
export interface BaseComponentProps {
  className?: string;
  style?: React.CSSProperties;
  children?: React.ReactNode;
}

/** 数据可视化组件Props */
export interface VisualizationProps extends BaseComponentProps {
  data: any[];
  config: ChartConfig;
  loading?: boolean;
  error?: string;
  onDataPointClick?: (data: any) => void;
}

/** 认知状态显示组件Props */
export interface CognitiveDisplayProps extends BaseComponentProps {
  cognitiveState: CognitiveState;
  historical?: CognitiveStateHistory;
  realtime?: boolean;
  showDetails?: boolean;
}

// ============ 工具类型 ============

/** 深度可选类型 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/** 排除某些属性的类型 */
export type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

/** 提取Promise的返回类型 */
export type PromiseReturnType<T> = T extends Promise<infer R> ? R : T;