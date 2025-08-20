/**
 * API 基础服务
 * 提供统一的HTTP请求处理、错误处理、拦截器等功能
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { API_CONFIG, API_ENDPOINTS, ERROR_CODES, STORAGE_KEYS } from '@/constants';
import { ApiResponse } from '@/types';

// 创建axios实例
const apiClient: AxiosInstance = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: API_CONFIG.TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============ 请求拦截器 ============

apiClient.interceptors.request.use(
  (config) => {
    // 添加认证token
    const token = localStorage.getItem(STORAGE_KEYS.ACCESS_TOKEN);
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // 添加请求ID用于追踪
    config.headers['X-Request-ID'] = generateRequestId();
    
    // 添加时间戳
    config.headers['X-Timestamp'] = Date.now().toString();
    
    // 开发模式下记录请求
    if (process.env.NODE_ENV === 'development') {
      console.log('🚀 API Request:', {
        method: config.method?.toUpperCase(),
        url: config.url,
        data: config.data,
        headers: config.headers,
      });
    }
    
    return config;
  },
  (error) => {
    console.error('❌ Request Error:', error);
    return Promise.reject(error);
  }
);

// ============ 响应拦截器 ============

apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    // 开发模式下记录响应
    if (process.env.NODE_ENV === 'development') {
      console.log('✅ API Response:', {
        status: response.status,
        url: response.config.url,
        data: response.data,
      });
    }
    
    return response;
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };
    
    // 开发模式下记录错误
    if (process.env.NODE_ENV === 'development') {
      console.error('❌ API Error:', {
        status: error.response?.status,
        url: error.config?.url,
        message: error.message,
        data: error.response?.data,
      });
    }
    
    // 401错误处理 - Token过期
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        const refreshToken = localStorage.getItem(STORAGE_KEYS.REFRESH_TOKEN);
        if (refreshToken) {
          // 尝试刷新token
          const response = await refreshTokenRequest(refreshToken);
          const newToken = response.data.accessToken;
          
          // 更新存储的token
          localStorage.setItem(STORAGE_KEYS.ACCESS_TOKEN, newToken);
          
          // 重新发送原请求
          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${newToken}`;
          }
          
          return apiClient(originalRequest);
        }
      } catch (refreshError) {
        // 刷新失败，清除所有认证信息并跳转到登录页
        localStorage.removeItem(STORAGE_KEYS.ACCESS_TOKEN);
        localStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN);
        localStorage.removeItem(STORAGE_KEYS.USER_INFO);
        
        // 这里可以触发全局的登录状态重置
        window.location.href = '/login';
        
        return Promise.reject(refreshError);
      }
    }
    
    // 网络错误处理
    if (!error.response) {
      return Promise.reject({
        code: ERROR_CODES.NETWORK_ERROR,
        message: '网络连接失败，请检查网络设置',
        originalError: error,
      });
    }
    
    // 其他HTTP错误处理
    const errorResponse = error.response.data || {};
    return Promise.reject({
      code: errorResponse.code || getErrorCodeByStatus(error.response.status),
      message: errorResponse.message || getErrorMessageByStatus(error.response.status),
      status: error.response.status,
      originalError: error,
    });
  }
);

// ============ 工具函数 ============

/** 生成请求ID */
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/** 刷新Token请求 */
async function refreshTokenRequest(refreshToken: string) {
  return axios.post(
    `${API_CONFIG.BASE_URL}${API_ENDPOINTS.AUTH.REFRESH}`,
    { refreshToken },
    { timeout: API_CONFIG.TIMEOUT }
  );
}

/** 根据HTTP状态码获取错误码 */
function getErrorCodeByStatus(status: number): string {
  switch (status) {
    case 400:
      return ERROR_CODES.INVALID_DATA;
    case 401:
      return ERROR_CODES.UNAUTHORIZED;
    case 403:
      return ERROR_CODES.FORBIDDEN;
    case 404:
      return ERROR_CODES.DATA_NOT_FOUND;
    case 408:
      return ERROR_CODES.TIMEOUT_ERROR;
    case 409:
      return ERROR_CODES.DUPLICATE_DATA;
    case 500:
      return ERROR_CODES.INTERNAL_ERROR;
    case 503:
      return ERROR_CODES.SERVICE_UNAVAILABLE;
    default:
      return ERROR_CODES.INTERNAL_ERROR;
  }
}

/** 根据HTTP状态码获取错误信息 */
function getErrorMessageByStatus(status: number): string {
  switch (status) {
    case 400:
      return '请求数据格式错误';
    case 401:
      return '未授权访问，请重新登录';
    case 403:
      return '权限不足，无法访问';
    case 404:
      return '请求的资源不存在';
    case 408:
      return '请求超时，请稍后重试';
    case 409:
      return '数据冲突，请检查后重试';
    case 500:
      return '服务器内部错误';
    case 503:
      return '服务暂时不可用，请稍后重试';
    default:
      return '发生未知错误';
  }
}

// ============ 重试机制 ============

/** 带重试的请求 */
async function requestWithRetry<T>(
  requestConfig: AxiosRequestConfig,
  retryTimes: number = API_CONFIG.RETRY_TIMES
): Promise<T> {
  let lastError: any;
  
  for (let i = 0; i <= retryTimes; i++) {
    try {
      const response = await apiClient(requestConfig);
      return response.data;
    } catch (error: any) {
      lastError = error;
      
      // 如果是认证错误或客户端错误，不重试
      if (error.status && error.status >= 400 && error.status < 500) {
        throw error;
      }
      
      // 最后一次重试失败
      if (i === retryTimes) {
        throw error;
      }
      
      // 等待后重试
      const delay = API_CONFIG.RETRY_DELAY * Math.pow(2, i); // 指数退避
      await new Promise(resolve => setTimeout(resolve, delay));
      
      console.warn(`Request failed, retrying... (${i + 1}/${retryTimes})`);
    }
  }
  
  throw lastError;
}

// ============ API方法封装 ============

/** GET请求 */
export async function get<T = any>(
  url: string,
  params?: any,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> {
  const requestConfig: AxiosRequestConfig = {
    method: 'GET',
    url,
    params,
    ...config,
  };
  
  return requestWithRetry<ApiResponse<T>>(requestConfig);
}

/** POST请求 */
export async function post<T = any>(
  url: string,
  data?: any,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> {
  const requestConfig: AxiosRequestConfig = {
    method: 'POST',
    url,
    data,
    ...config,
  };
  
  return requestWithRetry<ApiResponse<T>>(requestConfig);
}

/** PUT请求 */
export async function put<T = any>(
  url: string,
  data?: any,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> {
  const requestConfig: AxiosRequestConfig = {
    method: 'PUT',
    url,
    data,
    ...config,
  };
  
  return requestWithRetry<ApiResponse<T>>(requestConfig);
}

/** DELETE请求 */
export async function del<T = any>(
  url: string,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> {
  const requestConfig: AxiosRequestConfig = {
    method: 'DELETE',
    url,
    ...config,
  };
  
  return requestWithRetry<ApiResponse<T>>(requestConfig);
}

/** PATCH请求 */
export async function patch<T = any>(
  url: string,
  data?: any,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> {
  const requestConfig: AxiosRequestConfig = {
    method: 'PATCH',
    url,
    data,
    ...config,
  };
  
  return requestWithRetry<ApiResponse<T>>(requestConfig);
}

/** 文件上传 */
export async function upload<T = any>(
  url: string,
  file: File | FormData,
  onProgress?: (progressEvent: any) => void,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> {
  const formData = file instanceof FormData ? file : new FormData();
  
  if (file instanceof File) {
    formData.append('file', file);
  }
  
  const requestConfig: AxiosRequestConfig = {
    method: 'POST',
    url,
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: onProgress,
    ...config,
  };
  
  // 文件上传不使用重试机制
  const response = await apiClient(requestConfig);
  return response.data;
}

/** 文件下载 */
export async function download(
  url: string,
  filename?: string,
  config?: AxiosRequestConfig
): Promise<void> {
  const requestConfig: AxiosRequestConfig = {
    method: 'GET',
    url,
    responseType: 'blob',
    ...config,
  };
  
  const response = await apiClient(requestConfig);
  
  // 创建下载链接
  const blob = new Blob([response.data]);
  const downloadUrl = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = filename || 'download';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(downloadUrl);
}

// ============ 批量请求 ============

/** 并发请求 */
export async function concurrent<T = any>(
  requests: AxiosRequestConfig[]
): Promise<ApiResponse<T>[]> {
  const promises = requests.map(config => apiClient(config));
  const responses = await Promise.allSettled(promises);
  
  return responses.map((result, index) => {
    if (result.status === 'fulfilled') {
      return result.value.data;
    } else {
      console.error(`Request ${index} failed:`, result.reason);
      throw result.reason;
    }
  });
}

/** 顺序请求 */
export async function sequential<T = any>(
  requests: AxiosRequestConfig[]
): Promise<ApiResponse<T>[]> {
  const results: ApiResponse<T>[] = [];
  
  for (const config of requests) {
    try {
      const response = await apiClient(config);
      results.push(response.data);
    } catch (error) {
      console.error('Sequential request failed:', error);
      throw error;
    }
  }
  
  return results;
}

// ============ 缓存机制 ============

const requestCache = new Map<string, { data: any; timestamp: number; ttl: number }>();

/** 带缓存的GET请求 */
export async function getWithCache<T = any>(
  url: string,
  params?: any,
  ttl: number = 5 * 60 * 1000, // 默认5分钟缓存
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> {
  const cacheKey = `${url}?${new URLSearchParams(params).toString()}`;
  const cached = requestCache.get(cacheKey);
  
  // 检查缓存是否有效
  if (cached && Date.now() - cached.timestamp < cached.ttl) {
    return cached.data;
  }
  
  // 发起请求
  const response = await get<T>(url, params, config);
  
  // 缓存响应
  requestCache.set(cacheKey, {
    data: response,
    timestamp: Date.now(),
    ttl,
  });
  
  return response;
}

/** 清除缓存 */
export function clearCache(pattern?: string): void {
  if (pattern) {
    // 清除匹配模式的缓存
    for (const key of requestCache.keys()) {
      if (key.includes(pattern)) {
        requestCache.delete(key);
      }
    }
  } else {
    // 清除所有缓存
    requestCache.clear();
  }
}

// ============ 请求取消 ============

const cancelTokens = new Map<string, AbortController>();

/** 可取消的请求 */
export async function cancellableRequest<T = any>(
  requestConfig: AxiosRequestConfig,
  cancelKey?: string
): Promise<ApiResponse<T>> {
  const key = cancelKey || generateRequestId();
  
  // 创建取消控制器
  const controller = new AbortController();
  cancelTokens.set(key, controller);
  
  try {
    const response = await apiClient({
      ...requestConfig,
      signal: controller.signal,
    });
    
    return response.data;
  } finally {
    // 清理取消控制器
    cancelTokens.delete(key);
  }
}

/** 取消请求 */
export function cancelRequest(cancelKey: string): void {
  const controller = cancelTokens.get(cancelKey);
  if (controller) {
    controller.abort();
    cancelTokens.delete(cancelKey);
  }
}

/** 取消所有请求 */
export function cancelAllRequests(): void {
  for (const [key, controller] of cancelTokens.entries()) {
    controller.abort();
    cancelTokens.delete(key);
  }
}

// 导出axios实例供其他模块使用
export { apiClient };
export default apiClient;