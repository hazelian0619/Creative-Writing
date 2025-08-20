#!/bin/bash

# Team B - 认知状态后端服务部署脚本
# 快速部署和管理脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目配置
PROJECT_NAME="cognitive-state-backend"
DOCKER_COMPOSE_FILE="docker/docker-compose.yml"
ENV_FILE=".env"

# 函数定义
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "检查部署要求..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查Team A依赖
    if [ ! -d "team_a" ]; then
        log_error "未找到team_a目录，请确保Team A的代码在正确位置"
        exit 1
    fi
    
    log_info "✅ 依赖检查完成"
}

setup_environment() {
    log_info "设置环境配置..."
    
    # 创建.env文件如果不存在
    if [ ! -f "$ENV_FILE" ]; then
        log_info "创建环境配置文件..."
        cp .env.example "$ENV_FILE"
        log_warn "请检查并修改 $ENV_FILE 中的配置"
    fi
    
    # 创建必要的目录
    mkdir -p logs
    mkdir -p docker/mongodb/data
    mkdir -p docker/prometheus/data
    mkdir -p docker/grafana/data
    
    # 设置权限
    chmod 755 logs
    
    log_info "✅ 环境设置完成"
}

build_images() {
    log_info "构建Docker镜像..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    
    log_info "✅ 镜像构建完成"
}

start_services() {
    log_info "启动服务..."
    
    # 启动基础服务
    log_info "启动基础服务（MongoDB, Redis）..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d mongodb redis
    
    # 等待数据库启动
    log_info "等待数据库启动..."
    sleep 10
    
    # 启动应用服务
    log_info "启动应用服务..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    log_info "✅ 所有服务已启动"
}

stop_services() {
    log_info "停止服务..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    log_info "✅ 服务已停止"
}

restart_services() {
    log_info "重启服务..."
    
    stop_services
    start_services
}

show_status() {
    log_info "服务状态："
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    echo ""
    log_info "服务健康检查："
    
    # 检查API健康状态
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log_info "✅ API服务健康"
    else
        log_error "❌ API服务不健康"
    fi
    
    # 检查Prometheus
    if curl -f http://localhost:9091 >/dev/null 2>&1; then
        log_info "✅ Prometheus监控健康"
    else
        log_error "❌ Prometheus监控不健康"
    fi
    
    # 检查Grafana
    if curl -f http://localhost:3000 >/dev/null 2>&1; then
        log_info "✅ Grafana仪表板健康"
    else
        log_error "❌ Grafana仪表板不健康"
    fi
}

show_logs() {
    local service=${1:-}
    
    if [ -n "$service" ]; then
        log_info "显示 $service 服务日志："
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f "$service"
    else
        log_info "显示所有服务日志："
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
    fi
}

run_tests() {
    log_info "运行测试..."
    
    # 等待服务启动
    log_info "等待服务启动完成..."
    sleep 15
    
    # 运行集成测试
    log_info "运行API集成测试..."
    python -m pytest tests/integration/ -v
    
    # 运行性能测试
    log_info "运行性能测试..."
    python -m pytest tests/performance/ -v
    
    log_info "✅ 测试完成"
}

cleanup() {
    log_info "清理资源..."
    
    # 停止并删除容器
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v
    
    # 删除镜像（可选）
    if [ "$1" = "--images" ]; then
        log_info "删除Docker镜像..."
        docker images | grep cognitive | awk '{print $3}' | xargs -r docker rmi
    fi
    
    # 清理日志
    if [ "$1" = "--logs" ]; then
        log_info "清理日志文件..."
        rm -rf logs/*
    fi
    
    log_info "✅ 清理完成"
}

backup_data() {
    log_info "备份数据..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # 备份MongoDB数据
    log_info "备份MongoDB数据..."
    docker exec cognitive-mongodb mongodump --out /tmp/backup
    docker cp cognitive-mongodb:/tmp/backup "$BACKUP_DIR/mongodb"
    
    # 备份配置文件
    log_info "备份配置文件..."
    cp -r docker "$BACKUP_DIR/"
    cp "$ENV_FILE" "$BACKUP_DIR/"
    
    log_info "✅ 数据备份完成：$BACKUP_DIR"
}

show_help() {
    echo "认知状态后端服务部署脚本"
    echo ""
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令："
    echo "  start       启动所有服务"
    echo "  stop        停止所有服务"
    echo "  restart     重启所有服务"
    echo "  status      显示服务状态"
    echo "  logs [服务] 显示日志"
    echo "  test        运行测试"
    echo "  build       构建Docker镜像"
    echo "  cleanup     清理资源"
    echo "  backup      备份数据"
    echo "  help        显示帮助"
    echo ""
    echo "示例："
    echo "  $0 start                    # 启动所有服务"
    echo "  $0 logs cognitive-backend   # 查看后端服务日志"
    echo "  $0 cleanup --images         # 清理包括镜像"
    echo ""
}

main() {
    case "${1:-}" in
        start)
            check_requirements
            setup_environment
            start_services
            sleep 5
            show_status
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            sleep 5
            show_status
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "$2"
            ;;
        test)
            run_tests
            ;;
        build)
            check_requirements
            build_images
            ;;
        cleanup)
            cleanup "$2"
            ;;
        backup)
            backup_data
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: ${1:-}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 检查是否以root权限运行
if [ "$EUID" -eq 0 ]; then
    log_warn "不建议以root权限运行此脚本"
fi

# 执行主函数
main "$@"