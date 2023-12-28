#![no_std]

//#![feature(fn_traits)]
//#![feature(type_alias_impl_trait)]

use core::{marker::PhantomData, mem, ptr};
use void::Void;

pub trait H1 {
    type H<A>;
}

pub trait H2 {
    type H<A, B>;
}

pub trait EndofunctorOnce: EndofunctorMut {
    fn map_once<A, B>(_: impl FnOnce(A) -> B, _: Self::H<A>) -> Self::H<B>;
}

pub trait EndofunctorMut: Endofunctor {
    fn map_mut<A, B>(_: impl FnMut(A) -> B, _: Self::H<A>) -> Self::H<B>;
}

pub trait Endofunctor: H1 {
    fn map<A, B>(_: impl Fn(A) -> B, _: Self::H<A>) -> Self::H<B>;
}

pub trait ApplicableOnce: ApplicableMut + EndofunctorOnce {
    #[inline]
    fn ap_once<A, B>(f: Self::H<impl FnOnce(A) -> B>, a: Self::H<A>) -> Self::H<B> { Self::liftA2_once(|f, a| f(a), f, a) }

    #[inline]
    fn liftA2_once<A, B, C>(f: impl FnOnce(A, B) -> C, a: Self::H<A>, b: Self::H<B>) -> Self::H<C> {
        Self::ap_once(Self::map_once(move |a| |b| f(a, b), a), b)
    }
}

pub trait ApplicableMut: Applicable + EndofunctorMut {
    #[inline]
    fn ap_mut<A, B>(f: Self::H<impl FnMut(A) -> B>, a: Self::H<A>) -> Self::H<B> { Self::liftA2_mut(|mut f, a| f(a), f, a) }

    fn liftA2_mut<A, B, C>(_: impl FnMut(A, B) -> C, _: Self::H<A>, _: Self::H<B>) -> Self::H<C>;
}

pub trait Applicable: Endofunctor {
    #[inline]
    fn ap<A, B>(f: Self::H<impl Fn(A) -> B>, a: Self::H<A>) -> Self::H<B> { Self::liftA2(|f, a| f(a), f, a) }

    fn liftA2<A, B, C>(_: impl Fn(A, B) -> C, _: Self::H<A>, _: Self::H<B>) -> Self::H<C>;
}

pub trait Semimonad: Endofunctor {
    fn join<A>(_: Self::H<Self::H<A>>) -> Self::H<A>;
}

#[inline]
pub fn bind_once<A, B, F: Semimonad + EndofunctorOnce, Φ: FnOnce(A) -> F::H<B>>(f: Φ, a: F::H<A>) -> F::H<B> { F::join(F::map_once(f, a)) }

#[inline]
pub fn bind_mut<A, B, F: Semimonad + EndofunctorMut, Φ: FnMut(A) -> F::H<B>>(f: Φ, a: F::H<A>) -> F::H<B> { F::join(F::map_mut(f, a)) }

#[inline]
pub fn bind<A, B, F: Semimonad + EndofunctorMut, Φ: Fn(A) -> F::H<B>>(f: Φ, a: F::H<A>) -> F::H<B> { F::join(F::map(f, a)) }

pub trait Pointed: H1 {
    fn point<A>(_: A) -> Self::H<A>;
}

pub trait TraversableOnce: EndofunctorOnce + TraversableMut {
    fn traverse_once<A, B, P: Pointed + ApplicableOnce>(_: impl FnOnce(A) -> P::H<B>, _: Self::H<A>) -> P::H<Self::H<B>>;
}

pub trait TraversableMut: EndofunctorMut + Traversable {
    fn traverse_mut<A, B, P: Pointed + ApplicableMut>(_: impl FnMut(A) -> P::H<B>, _: Self::H<A>) -> P::H<Self::H<B>>;
}

pub trait Traversable: Endofunctor {
    fn traverse<A, B, P: Pointed + Applicable>(_: impl Fn(A) -> P::H<B>, _: Self::H<A>) -> P::H<Self::H<B>>;
}

pub struct ArrayW<const N: usize>(Void);

impl<const N: usize> H1 for ArrayW<N> {
    type H<A> = [A; N];
}

impl<const N: usize> EndofunctorMut for ArrayW<N> {
    #[inline]
    fn map_mut<A, B>(mut f: impl FnMut(A) -> B, a: [A; N]) -> [B; N] { map_array_with_ix_mut(|a, _| f(a), a) }
}

impl<const N: usize> Endofunctor for ArrayW<N> {
    #[inline]
    fn map<A, B>(f: impl Fn(A) -> B, a: [A; N]) -> [B; N] { Self::map_mut(f, a) }
}

impl<const N: usize> ApplicableMut for ArrayW<N> {
    #[inline]
    fn liftA2_mut<A, B, C>(mut f: impl FnMut(A, B) -> C, a: [A; N], b: [B; N]) -> [C; N] { unsafe {
        let a = mem::ManuallyDrop::new(a);
        let b = mem::ManuallyDrop::new(b);
        let mut c = mem::MaybeUninit::<[C; N]>::uninit();
        for k in 0..N { ptr::write((c.as_mut_ptr() as *mut C).wrapping_add(k), f(ptr::read(&a[k]), ptr::read(&b[k]))) }
        c.assume_init()
    } }
}

impl<const N: usize> Applicable for ArrayW<N> {
    #[inline]
    fn liftA2<A, B, C>(f: impl Fn(A, B) -> C, a: [A; N], b: [B; N]) -> [C; N] { Self::liftA2_mut(f, a, b) }
}

impl<const N: usize> Semimonad for ArrayW<N> {
    #[inline]
    fn join<A>(a: [[A; N]; N]) -> [A; N] {
        map_array_with_ix_mut(|a, k| unsafe { let a = mem::ManuallyDrop::new(a); ptr::read(&a[k]) }, a)
    }
}

impl<const N: usize> TraversableMut for ArrayW<N> {
    #[inline]
    fn traverse_mut<A, B, P: Pointed + Applicable>(mut f: impl FnMut(A) -> P::H<B>, a: [A; N]) -> P::H<[B; N]> { traverse_array_with_ix_mut::<_, _, P, _, N>(|a, _| f(a), a) }
}

impl<const N: usize> Traversable for ArrayW<N> {
    #[inline]
    fn traverse<A, B, P: Pointed + Applicable>(f: impl Fn(A) -> P::H<B>, a: [A; N]) -> P::H<[B; N]> { traverse_array_with_ix_mut::<_, _, P, _, N>(|a, _| f(a), a) }
}

impl Pointed for ArrayW<1> {
    #[inline]
    fn point<A>(a: A) -> [A; 1] { [a] }
}

impl EndofunctorOnce for ArrayW<1> {
    #[inline]
    fn map_once<A, B>(f: impl FnOnce(A) -> B, [a]: [A; 1]) -> [B; 1] { [f(a)] }
}

#[inline]
fn map_array_with_ix_mut<A, B, const N: usize>(mut f: impl FnMut(A, usize) -> B, a: [A; N]) -> [B; N] { unsafe {
    let a = mem::ManuallyDrop::new(a);
    let mut b = mem::MaybeUninit::<[B; N]>::uninit();
    for k in 0..N { ptr::write((b.as_mut_ptr() as *mut B).wrapping_add(k), f(ptr::read(&a[k]), k)); }
    b.assume_init()
} }

#[inline]
fn traverse_array_with_ix_mut<A, B, P: Pointed + Applicable, F: FnMut(A, usize) -> P::H<B>, const N: usize>(mut f: F, a: [A; N]) -> P::H<[B; N]> { unsafe {
    let a = mem::ManuallyDrop::new(a);
    let mut bsp = P::point(mem::MaybeUninit::<[B; N]>::uninit());
    for k in 0..N {
        let bp = f(ptr::read(&a[k]), k);
        bsp = P::liftA2(|b, mut bs| {
            ptr::write((bs.as_mut_ptr() as *mut B).wrapping_add(k), b);
            bs
        }, bp, bsp);
    }
    P::map(|x| x.assume_init(), bsp)
} }

#[inline]
fn zip_arrays_with_ix_mut<A, B, C, F: FnMut(A, B, usize) -> C, const N: usize>(mut f: F, a: [A; N], b: [B; N]) -> [C; N] { unsafe {
        let a = mem::ManuallyDrop::new(a);
        let b = mem::ManuallyDrop::new(b);
    let mut c = mem::MaybeUninit::<[C; N]>::uninit();
    for k in 0..N { ptr::write((c.as_mut_ptr() as *mut C).wrapping_add(k), f(ptr::read(&a[k]), ptr::read(&b[k]), k)); }
    c.assume_init()
} }

#[allow(unused)]
#[inline]
fn zipA_arrays_with_ix_mut<A, B, C, P: Pointed + Applicable, F: FnMut(A, B, usize) -> P::H<C>, const N: usize>(mut f: F, a: [A; N], b: [B; N]) -> P::H<[C; N]> { unsafe {
    let a = mem::ManuallyDrop::new(a);
    let b = mem::ManuallyDrop::new(b);
    let mut csp = P::point(mem::MaybeUninit::<[C; N]>::uninit());
    for k in 0..N {
        let cp = f(ptr::read(&a[k]), ptr::read(&b[k]), k);
        csp = P::liftA2(|c, mut cs| {
            ptr::write((cs.as_mut_ptr() as *mut C).wrapping_add(k), c);
            cs
        }, cp, csp);
    }
    P::map(|x| x.assume_init(), csp)
} }

pub struct OptionW(Void);

impl H1 for OptionW {
    type H<A> = Option<A>;
}

impl EndofunctorOnce for OptionW {
    #[inline]
    fn map_once<A, B>(f: impl FnOnce(A) -> B, a: Option<A>) -> Option<B> { a.map(f) }
}

impl EndofunctorMut for OptionW {
    #[inline]
    fn map_mut<A, B>(f: impl FnMut(A) -> B, a: Option<A>) -> Option<B> { a.map(f) }
}

impl Endofunctor for OptionW {
    #[inline]
    fn map<A, B>(f: impl Fn(A) -> B, a: Option<A>) -> Option<B> { a.map(f) }
}

impl ApplicableOnce for OptionW {
    #[inline]
    fn liftA2_once<A, B, C>(f: impl FnOnce(A, B) -> C, a: Option<A>, b: Option<B>) -> Option<C> { match (a, b) {
        (Some(a), Some(b)) => Some(f(a, b)),
        _ => None,
    } }
}

impl ApplicableMut for OptionW {
    #[inline]
    fn liftA2_mut<A, B, C>(f: impl FnMut(A, B) -> C, a: Option<A>, b: Option<B>) -> Option<C> { Self::liftA2_once(f, a, b) }
}

impl Applicable for OptionW {
    #[inline]
    fn liftA2<A, B, C>(f: impl Fn(A, B) -> C, a: Option<A>, b: Option<B>) -> Option<C> { Self::liftA2_once(f, a, b) }
}

impl Semimonad for OptionW {
    #[inline]
    fn join<A>(a: Option<Option<A>>) -> Option<A> { match a { Some(a) => a, None => None, } }
}

impl TraversableOnce for OptionW {
    #[inline]
    fn traverse_once<A, B, P: Pointed + ApplicableOnce>(f: impl FnOnce(A) -> P::H<B>, a: Option<A>) -> P::H<Option<B>> { match a {
        None => P::point(None),
        Some(a) => P::map(Some, f(a)),
    } }
}

impl TraversableMut for OptionW {
    #[inline]
    fn traverse_mut<A, B, P: Pointed + ApplicableMut>(mut f: impl FnMut(A) -> P::H<B>, a: Option<A>) -> P::H<Option<B>> { match a {
        None => P::point(None),
        Some(a) => P::map(Some, f(a)),
    } }
}

impl Traversable for OptionW {
    #[inline]
    fn traverse<A, B, P: Pointed + Applicable>(f: impl Fn(A) -> P::H<B>, a: Option<A>) -> P::H<Option<B>> { match a {
        None => P::point(None),
        Some(a) => P::map(Some, f(a)),
    } }
}

impl Pointed for OptionW {
    #[inline]
    fn point<A>(a: A) -> Option<A> { Some(a) }
}

pub struct ResultW<E>(PhantomData<E>, Void);

impl<E> H1 for ResultW<E> {
    type H<A> = Result<A, E>;
}

impl<E> EndofunctorOnce for ResultW<E> {
    #[inline]
    fn map_once<A, B>(f: impl FnOnce(A) -> B, a: Result<A, E>) -> Result<B, E> { a.map(f) }
}

impl<E> EndofunctorMut for ResultW<E> {
    #[inline]
    fn map_mut<A, B>(f: impl FnMut(A) -> B, a: Result<A, E>) -> Result<B, E> { a.map(f) }
}

impl<E> Endofunctor for ResultW<E> {
    #[inline]
    fn map<A, B>(f: impl Fn(A) -> B, a: Result<A, E>) -> Result<B, E> { a.map(f) }
}

impl<E: Semigroup> ApplicableOnce for ResultW<E> {
    #[inline]
    fn liftA2_once<A, B, C>(f: impl FnOnce(A, B) -> C, a: Result<A, E>, b: Result<B, E>) -> Result<C, E> { match (a, b) {
        (Ok(a), Ok(b)) => Ok(f(a, b)),
        (Ok(_), Err(y)) => Err(y),
        (Err(x), Ok(_)) => Err(x),
        (Err(x), Err(y)) => Err(E::combine(x, y)),
    } }
}

impl<E: Semigroup> ApplicableMut for ResultW<E> {
    #[inline]
    fn liftA2_mut<A, B, C>(f: impl FnMut(A, B) -> C, a: Result<A, E>, b: Result<B, E>) -> Result<C, E> { Self::liftA2_once(f, a, b) }
}

impl<E: Semigroup> Applicable for ResultW<E> {
    #[inline]
    fn liftA2<A, B, C>(f: impl Fn(A, B) -> C, a: Result<A, E>, b: Result<B, E>) -> Result<C, E> { Self::liftA2_once(f, a, b) }
}

impl<E> Pointed for ResultW<E> {
    #[inline]
    fn point<A>(a: A) -> Result<A, E> { Ok(a) }
}

pub trait Semigroup {
    fn combine(_: Self, _: Self) -> Self;
}

impl<A: Semigroup, const N: usize> Semigroup for [A; N] {
    #[inline]
    fn combine(a: Self, b: Self) -> Self { zip_arrays_with_ix_mut(|a, b, _| A::combine(a, b), a, b) }
}

// We not define `Monoid`, for we can rather use `Default + Semigroup`.

macro_rules! impl_Semigroup_etc_tuple {
    ($($A:ident, $n:tt),*) => {
        impl<$($A: Semigroup),*> Semigroup for ($($A),*) {
            #[allow(unused_variables)]
            #[inline]
            fn combine(a: Self, b: Self) -> Self { ($($A::combine(a.$n, b.$n)),*) }
        }
    }
}

/*
pub trait Group: Default + Semigroup {
    fn invert(_: Self) -> Self;
}

// stupid standard library has no general impl Default for arrays.

impl<A: Group, const N: usize> Group for [A; N] {
    #[inline]
    fn invert(a: Self) -> Self { map_array_with_ix_mut(|a, _| A::invert(a), a) }
}
*/

impl_Semigroup_etc_tuple!();
impl_Semigroup_etc_tuple!(A, 0, B, 1);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4, F, 5);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4, F, 5, G, 6);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4, F, 5, G, 6, H, 7);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4, F, 5, G, 6, H, 7, I, 8);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4, F, 5, G, 6, H, 7, I, 8, J, 9);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4, F, 5, G, 6, H, 7, I, 8, J, 9, K, 10);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4, F, 5, G, 6, H, 7, I, 8, J, 9, K, 10, L, 11);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4, F, 5, G, 6, H, 7, I, 8, J, 9, K, 10, L, 11, M, 12);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4, F, 5, G, 6, H, 7, I, 8, J, 9, K, 10, L, 11, M, 12, N, 13);
impl_Semigroup_etc_tuple!(A, 0, B, 1, C, 2, D, 3, E, 4, F, 5, G, 6, H, 7, I, 8, J, 9, K, 10, L, 11, M, 12, N, 13, O, 14);

/// Sugar for doing monadically
///
/// Example:
/// ```
/// use category::rust::ArrayW;
/// use category::rust::Pointed;
/// (category::monadically_once! { ArrayW<1>:
///     x <- [()];
///     [x]
/// })[0]
/// ```
/// ```
/// use category::rust::OptionW;
/// use category::rust::Pointed;
/// Option::unwrap_or(category::monadically_once! { OptionW:
///     x <- Some(());
///     _ <- None;
///     Some(category::rust::Semigroup::combine(x, y))
/// }, ())
/// ```
#[macro_export]
macro_rules! monadically_once {
    // Alas, Rust won't accept `... $p:pat <- ...`, so we must copy the irrefutable pattern syntax here.
    ($t:ty: _ <- $($r:tt)*) => ($crate::monadically_once!($t: (_) <- $($r)*));
    ($t:ty: $(ref)? $(mut)? $v:ident <- $($r:tt)*) => ($crate::monadically_once!($t: ($(ref)? $(mut)? $v) <- $($r)*));
    ($t:ty: $p:path { $($e:tt)* } <- $($r:tt)*) => ($crate::monadically_once!($t: ($p { $($e)* }) <- $($r)*));
    ($t:ty: ($p:pat,) <- $($r:tt)*) => ($crate::monadically_once!($t: (($p,)) <- $($r)*));
    ($t:ty: ($($p:pat),+ $(,)?) <- $($r:tt)*) => ($crate::monadically_once!($t: (($($p),+)) <- $($r)*));
    ($t:ty: ($($p:pat,)* ..) <- $($r:tt)*) => ($crate::monadically_once!($t: (($($p,)* ..)) <- $($r)*));
    ($t:ty: [$($p:pat),* $(,)?] <- $($r:tt)*) => ($crate::monadically_once!($t: ([$($p),*]) <- $($r)*));
    ($t:ty: [$($p:pat,)* ..] <- $($r:tt)*) => ($crate::monadically_once!($t: ([$($p,)* ..]) <- $($r)*));
    ($t:ty: ($p:pat) <- $x:expr; $($r:tt)*) => ($crate::bind_once::<_, _, $t, _>(move |$p| $crate::monadically_once!($t: $($r)*), $x));
    ($t:ty: $x:expr; $($r:tt)*) => ($crate::monadically_once!($t: _ <- $t; $($r)*));
    ($t:ty: let $p:pat = $x:expr; $($r:tt)*) => ({ let $p = $x; $crate::monadically_once!($t: $($r)*) });
    ($t:ty: $x:expr) => ($x);
}
