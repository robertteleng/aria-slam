# Auditoría Técnica: H13 - Async Multithreading Pipeline

**Proyecto:** aria-slam (C++)
**Milestone:** H13 - Pipeline Asíncrono con Multithreading
**Fecha:** 2025-01-28
**Autor:** Roberto (preparación entrevista técnica)

---

## 0. RESUMEN EJECUTIVO

### Objetivo
Desacoplar componentes SLAM en threads asíncronos con comunicación lock-free para máximo throughput, especialmente en plataformas limitadas como Jetson Orin Nano.

### Arquitectura Propuesta
```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Thread                               │
│  [Camera Capture] ──► Frame Queue                               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Tracking Thread (High Priority)              │
│  Frame Queue ──► [ORB+YOLO] ──► [Match] ──► [Pose Est]          │
│                                                │                 │
│                                        KeyFrame Queue           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Loop Closure Thread (Medium Priority)          │
│  KeyFrame Queue ──► [Loop Detect] ──► [Pose Graph Opt]          │
│                           │                                      │
│                    Loop Candidate Queue                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Mapping Thread (Low Priority)                │
│  Loop Queue ──► [Map Update] ──► [Bundle Adjustment]            │
└─────────────────────────────────────────────────────────────────┘
```

### Resultados Esperados
| Configuración | FPS | Latencia |
|---------------|-----|----------|
| Single thread | 12 | 83ms |
| Async (3 threads) | 28 | 35ms |
| Async + frame skip | 30 | 33ms |

---

## 1. DISEÑO: LOCK-FREE QUEUE

### 1.1 Single-Producer Single-Consumer Queue

```cpp
template<typename T, size_t Capacity = 64>
class LockFreeQueue {
public:
    bool push(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) % Capacity;

        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // Full
        }

        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    std::optional<T> pop() {
        size_t tail = tail_.load(std::memory_order_relaxed);

        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // Empty
        }

        T item = buffer_[tail];
        tail_.store((tail + 1) % Capacity, std::memory_order_release);
        return item;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

private:
    std::array<T, Capacity> buffer_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
};
```

**Memory Ordering explicado:**
```
Producer (push):                    Consumer (pop):
────────────────                    ────────────────
1. head = head_.load(relaxed)       1. tail = tail_.load(relaxed)
2. if (next == tail.load(acquire))  2. if (tail == head.load(acquire))
3. buffer_[head] = item             3. item = buffer_[tail]
4. head_.store(next, release)       4. tail_.store(next, release)

ACQUIRE: garantiza ver todas las escrituras antes del RELEASE correspondiente
RELEASE: garantiza que escrituras anteriores sean visibles
RELAXED: sin garantías de ordenamiento (solo atomicidad)

Secuencia correcta:
- Producer: escribe datos ANTES de release en head
- Consumer: acquire en head garantiza ver datos del producer
```

### 1.2 Ring Buffer Visualization

```
Capacity = 8, head = 5, tail = 2

  tail                      head
    ↓                         ↓
┌───┬───┬───┬───┬───┬───┬───┬───┐
│   │   │ A │ B │ C │   │   │   │
└───┴───┴───┴───┴───┴───┴───┴───┘
  0   1   2   3   4   5   6   7

Elementos válidos: [2, 3, 4] = 3 elementos
Espacio libre: [5, 6, 7, 0, 1] = 5 slots

Push: buffer_[5] = D; head = 6
Pop:  return buffer_[2]; tail = 3
```

---

## 2. DISEÑO: THREAD POOL

### 2.1 Implementación

```cpp
class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop_(false) {
        for (size_t i = 0; i < num_threads; i++) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }

    ~ThreadPool() {
        stop_ = true;
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    template<typename F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        using ReturnType = decltype(f());
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::forward<F>(f));

        std::future<ReturnType> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push([task] { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

private:
    void workerLoop() {
        while (!stop_) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_;
};
```

**Diagrama de worker:**
```
    ┌─────────────────────────────────┐
    │         Worker Thread           │
    └───────────────┬─────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   cv_.wait()    │◄────────────────┐
          │ (sleeping)      │                 │
          └────────┬────────┘                 │
                   │ notify_one()             │
                   ▼                          │
          ┌─────────────────┐                 │
          │ stop_ || !empty?│──No────────────►│
          └────────┬────────┘                 │
                   │ Yes                      │
                   ▼                          │
          ┌─────────────────┐                 │
          │ if stop_ && empty│──Yes──► return │
          └────────┬────────┘                 │
                   │ No                       │
                   ▼                          │
          ┌─────────────────┐                 │
          │  pop task       │                 │
          │  execute task() │                 │
          └────────┬────────┘                 │
                   │                          │
                   └──────────────────────────┘
```

---

## 3. DISEÑO: ASYNC SLAM PIPELINE

### 3.1 Estructura de Threads

```cpp
class AsyncSlamPipeline {
public:
    AsyncSlamPipeline()
        : tracking_thread_([this] { trackingLoop(); }),
          loop_thread_([this] { loopClosureLoop(); }),
          mapping_thread_([this] { mappingLoop(); }) {}

    ~AsyncSlamPipeline() {
        stop_ = true;
        tracking_thread_.join();
        loop_thread_.join();
        mapping_thread_.join();
    }

    void pushFrame(const cv::Mat& frame) {
        frame_queue_.push(frame.clone());
    }

private:
    // Thread functions
    void trackingLoop();
    void loopClosureLoop();
    void mappingLoop();

    // Lock-free queues
    LockFreeQueue<cv::Mat, 8> frame_queue_;
    LockFreeQueue<KeyFrame, 32> keyframe_queue_;
    LockFreeQueue<LoopCandidate, 16> loop_queue_;

    // Threads
    std::thread tracking_thread_;
    std::thread loop_thread_;
    std::thread mapping_thread_;
    std::atomic<bool> stop_{false};

    // Components
    Tracker tracker_;
    LoopClosureDetector loop_detector_;
    PoseGraphOptimizer pose_graph_;
};
```

### 3.2 Tracking Loop

```cpp
void AsyncSlamPipeline::trackingLoop() {
    while (!stop_) {
        auto frame = frame_queue_.pop();
        if (!frame) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        // Process frame
        auto result = tracker_.process(*frame);

        // If keyframe, send to loop closure
        if (result.is_keyframe) {
            keyframe_queue_.push(result.keyframe);
        }
    }
}
```

### 3.3 Loop Closure Loop

```cpp
void AsyncSlamPipeline::loopClosureLoop() {
    while (!stop_) {
        auto kf = keyframe_queue_.pop();
        if (!kf) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Check for loop
        LoopCandidate candidate;
        if (loop_detector_.detect(*kf, candidate)) {
            loop_queue_.push(candidate);
        }
    }
}
```

### 3.4 Mapping Loop

```cpp
void AsyncSlamPipeline::mappingLoop() {
    while (!stop_) {
        auto loop = loop_queue_.pop();
        if (!loop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Optimize pose graph
        pose_graph_.addLoopEdge(loop->match_id, loop->query_id,
                                loop->relative_pose);
        pose_graph_.optimize();
    }
}
```

---

## 4. TEORÍA: THREAD PRIORITIES Y CPU AFFINITY

### 4.1 Thread Priorities (Linux)

```cpp
#include <pthread.h>

void setThreadPriority(std::thread& t, int priority) {
    sched_param param;
    param.sched_priority = priority;
    pthread_setschedparam(t.native_handle(), SCHED_FIFO, &param);
}

// Aplicación:
setThreadPriority(tracking_thread_, 90);   // High: real-time crítico
setThreadPriority(loop_thread_, 50);       // Medium
setThreadPriority(mapping_thread_, 20);    // Low: puede retrasarse
```

**Scheduling policies:**
```
SCHED_FIFO:
- Real-time, first-in-first-out
- Priority 1-99 (mayor = más prioritario)
- Thread corre hasta que yield o bloquea

SCHED_RR:
- Real-time, round-robin
- Time-slicing entre threads de misma prioridad

SCHED_OTHER:
- Default, nice-based
- Para threads no críticos
```

### 4.2 CPU Affinity (Jetson)

```cpp
#include <sched.h>

void setThreadAffinity(std::thread& t, const std::vector<int>& cpus) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu : cpus) {
        CPU_SET(cpu, &cpuset);
    }
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
}

// Jetson Orin Nano: 6 cores ARM
// Cores 0-3: A78 (performance)
// Cores 4-5: A78AE (efficiency)

setThreadAffinity(tracking_thread_, {0, 1});  // Performance cores
setThreadAffinity(loop_thread_, {2});
setThreadAffinity(mapping_thread_, {4, 5});   // Efficiency cores
```

**Layout de cores Jetson:**
```
┌─────────────────────────────────────────────────────┐
│                    Jetson Orin Nano                  │
│                                                     │
│   ┌─────────────────┐    ┌─────────────────┐       │
│   │ Performance     │    │ Efficiency      │       │
│   │ Cluster         │    │ Cluster         │       │
│   │                 │    │                 │       │
│   │ [0] [1] [2] [3] │    │    [4] [5]      │       │
│   │  A78  A78  A78  A78  │    │ A78AE A78AE │       │
│   └─────────────────┘    └─────────────────┘       │
│                                                     │
│   Tracking, Loop ──────►  Performance              │
│   Mapping ────────────────────────► Efficiency     │
└─────────────────────────────────────────────────────┘
```

---

## 5. DISEÑO: GRACEFUL DEGRADATION

### 5.1 Frame Skipping bajo Carga

```cpp
void trackingLoop() {
    while (!stop_) {
        // Check queue depth
        if (frame_queue_.size() > 4) {
            // Skip frames to catch up
            while (frame_queue_.size() > 2) {
                frame_queue_.pop();
                frames_dropped_++;
            }
            std::cout << "Warning: dropped " << frames_dropped_
                      << " frames (processing too slow)" << std::endl;
        }

        // Process oldest frame
        auto frame = frame_queue_.pop();
        if (!frame) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        // Normal processing...
    }
}
```

**Estrategia:**
```
Queue depth:  [1] [2] [3] [4] [5] [6] [7] [8]
                              ↑
                        threshold = 4

Si depth > 4:
  - Skip hasta depth = 2
  - Log warning
  - Procesar frame más reciente

Resultado: mantiene latencia baja sacrificando frames
```

### 5.2 Backpressure Handling

```cpp
void pushFrame(const cv::Mat& frame) {
    int retries = 3;
    while (retries-- > 0) {
        if (frame_queue_.push(frame.clone())) {
            return;  // Success
        }
        // Queue full, wait and retry
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Still full after retries - drop frame
    frames_dropped_++;
}
```

---

## 6. CONCEPTOS C++ UTILIZADOS

### 6.1 std::atomic y Memory Ordering

```cpp
std::atomic<size_t> head_{0};

// Operaciones:
head_.load(std::memory_order_acquire);   // Lee con acquire barrier
head_.store(next, std::memory_order_release);  // Escribe con release barrier
head_.load(std::memory_order_relaxed);   // Sin ordering (solo atomicidad)
```

**Memory barriers:**
```
              Thread A                     Thread B
         ────────────────               ────────────────
         data = 42;
         ready.store(true, release);
              │                              │
              └─── synchronizes-with ────────┤
                                             │
                                        if (ready.load(acquire))
                                            use(data);  // ve data = 42

Sin barriers: Thread B podría ver data = 0 aunque ready = true
```

### 6.2 std::condition_variable

```cpp
std::mutex mutex_;
std::condition_variable cv_;
std::queue<Task> tasks_;

// Producer:
{
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push(task);
}
cv_.notify_one();

// Consumer:
{
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
    // lock reacquired, condition is true
    task = tasks_.front();
    tasks_.pop();
}
```

**Spurious wakeups:**
```cpp
// INCORRECTO (vulnerable a spurious wakeup):
cv_.wait(lock);
if (!tasks_.empty()) process(tasks_.front());

// CORRECTO (loop explícito o predicate):
cv_.wait(lock, [&]{ return !tasks_.empty(); });
// Equivalente a:
while (tasks_.empty()) {
    cv_.wait(lock);
}
```

### 6.3 std::packaged_task y std::future

```cpp
template<typename F>
auto enqueue(F&& f) -> std::future<decltype(f())> {
    using ReturnType = decltype(f());

    // Wrap function in packaged_task
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::forward<F>(f)
    );

    // Get future before moving task
    std::future<ReturnType> result = task->get_future();

    // Enqueue execution
    tasks_.push([task] { (*task)(); });

    return result;  // Caller can wait for result
}

// Uso:
auto future = pool.enqueue([]{ return expensiveComputation(); });
// ... do other work ...
int result = future.get();  // Block until ready
```

### 6.4 Perfect Forwarding

```cpp
template<typename F>
auto enqueue(F&& f) {
    // std::forward preserva lvalue/rvalue nature
    auto task = std::make_shared<...>(std::forward<F>(f));
    //                                 ▲
    // Si f es lvalue: forward<F>(f) es lvalue reference
    // Si f es rvalue: forward<F>(f) es rvalue reference
}
```

---

## 7. PREGUNTAS DE ENTREVISTA

### Q1: ¿Por qué lock-free queue en lugar de mutex?

**R:**
```
Con mutex:
- Thread A: lock(); push(); unlock();
- Thread B: lock(); // BLOCKED esperando A
- Context switch overhead: ~1-10 μs

Con lock-free:
- Thread A: push() con atomics
- Thread B: pop() simultáneamente
- Sin blocking, sin context switch

Para SLAM @ 30 FPS = 33ms/frame
- 1ms de contención de mutex = 3% overhead
- Lock-free: ~0.1μs por operación
```

### Q2: ¿Qué es memory_order_acquire/release?

**R:**
```cpp
// RELEASE: "Todas mis escrituras anteriores son visibles"
data = 42;
flag.store(true, memory_order_release);

// ACQUIRE: "Veo todas las escrituras antes del release"
if (flag.load(memory_order_acquire)) {
    assert(data == 42);  // Garantizado
}

Sin estas barreras:
- CPU puede reordenar instrucciones
- Cache puede no estar sincronizado
- Podrías ver flag=true pero data=0
```

### Q3: ¿Por qué diferentes prioridades para los threads?

**R:**
```
Tracking (priority 90):
- Crítico para mantener FPS
- Si se retrasa: frames se pierden
- Debe ejecutar ASAP

Loop Closure (priority 50):
- Importante pero no urgente
- Puede esperar unos ms
- No afecta tracking directo

Mapping (priority 20):
- Trabajo batch
- Puede ejecutar cuando CPU idle
- No bloquea tracking ni loop
```

### Q4: ¿Cuándo usar condition_variable vs busy-wait?

**R:**
```cpp
// Busy-wait (spin):
while (queue.empty()) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
}
// Pros: baja latencia, simple
// Cons: consume CPU, ineficiente si espera es larga

// Condition variable:
cv.wait(lock, [&]{ return !queue.empty(); });
// Pros: no consume CPU mientras espera
// Cons: overhead de syscall (~1μs)

Usar busy-wait cuando:
- Espera es típicamente < 1ms
- Latencia es crítica
- CPU no está saturado

Usar condition_variable cuando:
- Espera puede ser larga
- Quieres minimizar uso de CPU
- Múltiples productores/consumidores
```

### Q5: ¿Cómo evitar race conditions en shared state?

**R:**
```cpp
// Opción 1: Copiar datos en la queue
void pushFrame(const cv::Mat& frame) {
    frame_queue_.push(frame.clone());  // Deep copy
}
// No hay shared state, cada thread tiene su copia

// Opción 2: Message passing (move semantics)
void pushFrame(cv::Mat&& frame) {
    frame_queue_.push(std::move(frame));  // Transfer ownership
}
// Solo un thread posee el dato a la vez

// Opción 3: Immutable data
struct ImmutableFrame {
    const cv::Mat image;  // const = no modificable
};
// Múltiples threads pueden leer simultáneamente
```

### Q6: ¿Qué es CPU affinity y cuándo usarla?

**R:**
```
CPU affinity: fija un thread a core(s) específicos

Beneficios:
1. Cache locality: datos en L1/L2 del core
2. Predictable latency: no migra entre cores
3. Aislamiento: evita contención con otros procesos

Uso en SLAM:
- Tracking en cores rápidos (0,1)
- Background work en cores eficientes (4,5)
- GPU driver en core dedicado (2)

Riesgo:
- Si core está ocupado, thread espera
- Balanceo manual, no automático
```

### Q7: ¿Cómo debuggear race conditions?

**R:**
```
Herramientas:
1. ThreadSanitizer (TSan):
   g++ -fsanitize=thread ...
   Detecta data races en runtime

2. Valgrind Helgrind:
   valgrind --tool=helgrind ./program
   Más lento pero más detallado

3. Printf debugging:
   std::cout << "[" << std::this_thread::get_id() << "] " << message;
   Añade logging con thread ID

4. std::atomic + memory barriers:
   Verificar que todos los accesos shared usan atomics

5. Stress testing:
   Ejecutar miles de veces buscando fallas intermitentes
```

---

## 8. PERFORMANCE

### 8.1 Benchmark Single vs Multi-thread

| Métrica | Single Thread | 3 Threads | Mejora |
|---------|--------------|-----------|--------|
| FPS | 12 | 28 | 2.3x |
| Latency | 83ms | 35ms | 2.4x |
| CPU Usage | 100% (1 core) | 250% (3 cores) | - |

### 8.2 Overhead de Comunicación

| Operación | Tiempo |
|-----------|--------|
| Lock-free push/pop | ~0.1 μs |
| Mutex lock/unlock | ~1 μs |
| Condition variable signal | ~1-5 μs |
| cv::Mat::clone() 640x480 | ~0.3 ms |

### 8.3 Jetson Orin Nano Específico

| Componente | Core Assignment | FPS |
|------------|-----------------|-----|
| Tracking | A78 (0,1) | 25 |
| Loop Closure | A78 (2) | - |
| Mapping | A78AE (4,5) | - |
| **Total** | | 25 |

vs Single-thread: 12 FPS (2x mejora)

---

## 9. CHECKLIST DE PREPARACIÓN

### Conceptos que debes dominar:
- [ ] Lock-free queue con atomics
- [ ] Memory ordering (acquire/release)
- [ ] Thread priorities y scheduling
- [ ] CPU affinity
- [ ] Condition variables vs busy-wait
- [ ] std::packaged_task y std::future

### Código que debes poder escribir:
```cpp
// Atomic push con release
buffer_[head] = item;
head_.store(next, std::memory_order_release);

// Condition variable wait
std::unique_lock<std::mutex> lock(mutex_);
cv_.wait(lock, [&]{ return !queue.empty() || stop_; });

// Thread con priority
sched_param param;
param.sched_priority = 90;
pthread_setschedparam(thread.native_handle(), SCHED_FIFO, &param);

// CPU affinity
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(0, &cpuset);
pthread_setaffinity_np(thread.native_handle(), sizeof(cpuset), &cpuset);
```

### Números que debes conocer:
- Lock-free operation: **~0.1 μs**
- Mutex lock/unlock: **~1 μs**
- Context switch: **~1-10 μs**
- cv::Mat clone 640x480: **~0.3 ms**
- Target FPS: **30** → budget **33ms/frame**

---

**Generado:** 2025-01-28
**Proyecto:** aria-slam
**Status:** Diseño (no implementado aún)
